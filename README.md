# Wyoming Faster Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [faster-whisper](https://github.com/guillaumekln/faster-whisper/) speech to text system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_whisper)

[Source](https://github.com/home-assistant/addons/tree/master/whisper)

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/OHF-Voice/wyoming-faster-whisper.git
cd wyoming-faster-whisper
script/setup
```

Run a server anyone can connect to:

```sh
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`
(but see [Distil-Whisper models are not compatible](#distil-whisper-models-are-not-compatible)
if you want name biasing)

**NOTE**: Models are downloaded to the first `--data-dir` directory.

## Server-side VAD Endpointing

By default, the server waits for the client to send `audio-stop` before it
transcribes. Set `--vad-endpointing` to a number of seconds to end the command
after that much Silero-detected silence and send the transcript immediately:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data \
    --vad-endpointing 0.7
```

When enabled, the Wyoming service advertises that it does not require external
VAD, so Home Assistant leaves endpointing to this server. Speech must be detected
before the silence timer starts; brief pauses reset when speech resumes, and a
command is allowed to run for at least one second. Audio received after the
endpoint is ignored until the client's eventual `audio-stop`.

This is independent of the existing VAD options. `--vad-filter` still controls
Faster Whisper's filtering during transcription, while `--vad-clip` still trims
the completed batch WAV before transcription. They can be used together with
`--vad-endpointing`; the only intentional interaction is replacing Home
Assistant's external endpointing while this option is enabled.

## Reported Languages

The Wyoming service advertises the languages the **configured backend** can
actually transcribe, not a fixed list. Under the default `--stt-library auto` the
backend is chosen per language, with faster-whisper as the fallback for anything
the specialized backends don't claim, so all of Whisper's languages are reported.
Pinning a backend narrows it:

| Backend | Languages |
| --- | --- |
| `faster-whisper`, `transformers`, `whisper-cpp` | Whisper's 100 (just `en` for a `.en` checkpoint) |
| `qwen3-asr` | 31 |
| `sherpa` | 25 (Parakeet TDT v3) |
| `sherpa --sherpa-streaming` | `de`, `en`, `es`, `fr` (the published Kroko models) |
| `funasr` | `en`, `ja`, `ko`, `yue`, `zh`, `zh-CN`, `zh-HK`, `zh-TW` (SenseVoice) |
| `onnx-asr` | `ru` (GigaAM) |

`--language` is checked against this list at startup, and a region-qualified code
is accepted everywhere — `en-US` selects the same backend and model as `en`, and
`zh-HK`/`zh_HK` stays Cantonese rather than falling back to Mandarin.

This matters because Home Assistant picks the pipeline's STT language out of this
list. A backend that over-reports gets handed languages it will transcribe as
gibberish; one that under-reports is unreachable for languages it handles well.

Codes are reported in the form Home Assistant matches against, which is why
Cantonese is advertised as **`zh-HK`** rather than only `yue`. Home Assistant
treats `yue` and `zh` as unrelated languages, and `home-assistant/intents` ships
`zh-CN`/`zh-HK`/`zh-TW` with no bare `zh` or `yue` — so a backend that says only
`yue` can never be reached by a Cantonese pipeline. Both Cantonese-capable
backends (`funasr`, `qwen3-asr`) advertise `zh-HK` and decode it as Cantonese,
while `zh-CN` and `zh-TW` still fall through to Mandarin.

## Biasing Toward Your Home Assistant Names

Whisper has never heard of your thermostat. "What's the temperature of the Ecobee?"
comes back as "What's the temperature of the incubi?" — the acoustics were fine,
the model just has no reason to think that word exists.

Given a long-lived access token, the server reads the names in your home over the
Home Assistant websocket API and feeds them to the model as a prompt, which fixes
exactly that class of error:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data \
    --hass-token "$TOKEN" --hass-api 'http://homeassistant.local:8123/api'
```

Requires the `hass` extra:

```sh
pip install 'wyoming-faster-whisper[hass]'
```

It collects the names of **conversation-exposed** entities and their aliases, plus
your area and floor names — the names a speaker can actually say. It also reads
which area each exposed entity is in, so the areas you can actually command rank
above the ones you can't. Nothing else is read, and no service is ever called.

The fetch is free in latency terms: it starts when the audio starts, while the
speaker is still talking, and the names are ready by the time the audio stops.
Home Assistant being slow or unreachable only costs freshness — the previous names
are used, or none at all, and the transcript still comes back.

| Option | Default | Purpose |
| --- | --- | --- |
| `--hass-token` | | Long-lived access token. Enables everything above. Use `WYO_WHISPER_HASS_TOKEN_FILE` to keep it off the command line (see [Environment Variables](#environment-variables)). |
| `--hass-api` | `http://homeassistant.local:8123/api` | Where to find Home Assistant. |
| `--hass-refresh-seconds` | `0` | Minimum seconds between refreshes. `0` refreshes every utterance, so a rename takes effect immediately. |
| `--hass-prompt-max-tokens` | `200` | Token budget for names. Whisper's hard cap is 223 and quality falls off before it. |
| `--hass-prompt-timeout` | `1.0` | How long to wait on an unfinished refresh before transcribing with the names already on hand. |

A large home has more names than the budget holds, so they are added in priority
order and cut off when it runs out:

1. **Areas and floors that hold an exposed entity** — said in nearly every command
   ("turn on the *office* lamp"), and known to be real targets because something
   in them can actually be commanded.
2. **Entity names in the domains people say out loud** — `light`, `switch`, `fan`,
   `media_player`, `climate`, `scene`, `todo`. These are the proper nouns that get
   misheard.
3. **The remaining areas and floors** — sayable, but with nothing exposed in them
   there is no command they can complete.
4. **The remaining entity names** — in a big home mostly sensors, which are
   hundreds in number and usually asked about by area ("the temperature in the
   office") rather than by their own name.

Aliases are not a tier of their own. An alias is what you say *instead of* the
name the integration gave the thing, so each one sits directly behind the name it
belongs to and shares its tier — a light's "standing light" is kept or dropped
along with the light, never left out while lower-priority names get in.

Edit `PRIORITY_DOMAINS` in `wyoming_faster_whisper/hass_api.py` to change what
lands in tier 2. Run with `--debug` to see how many names were dropped and the
exact prompt used. `--initial-prompt` still works and is kept at the front of the
prompt, ahead of anything discovered from Home Assistant.

This biases `faster-whisper` and `qwen3-asr`, the backends that take a prompt.
Others ignore it.

### Distil-Whisper models are not compatible

Distil-Whisper checkpoints (`Systran/faster-distil-whisper-*`, `distil-small.en`,
`distil-large-v3`, …) were distilled without previous-text conditioning, so a
prompt is at best wasted on them and at worst destroys the transcription. This
applies to `--initial-prompt` as much as to `--hass-token`; the server warns at
startup and sends the prompt anyway, since a very short one may be harmless.

Measured on the same clean commands, comparing no prompt against a 29-name
(~97-token) list:

| Model | With a prompt |
| --- | --- |
| `small.en` | Works as intended: `Natalie Sparkly` → `Natalie's Heart Light` |
| `distil-small.en` | Breaks from ~52 prompt tokens on: `avg_logprob` falls below -1.0, every temperature fails, output truncates (`Start a timer for 25 minutes` → `Start a timer.`) or loops (`Add Hot Dog, Hot Dog, Hot Dog, …`) |
| `distil-large-v3` | Inert. No collapse at any size, but no biasing either — `Ecobee` still comes back `EcoBe` with the name in the prompt |

Use a standard Whisper model to bias toward your names.

### Prompt cost on qwen3-asr

For `qwen3-asr` the prompt is not free: the model has to read it before it starts
decoding, at roughly 2.8ms per token, so a 50-name list can double the time for a
short command.

The default model avoids this. The prompt sits ahead of the audio in the chat
template, so its state depends only on the prompt and is computed once, then
reused for every later utterance. The layout is chosen from the files present, so
older model directories with `decoder_init`/`decoder_step` keep working as before
— pass `--model rhasspy/qwen3-asr-0.6b-onnx-int4` to use one.

Measured on a Pi 5 (4 threads, 3.2s command, 50 names):

| | split | merged (default) |
| --- | --- | --- |
| latency | 3.42s | 2.20s |
| peak RSS | 2.25 GB | 1.55 GB |
| on disk | 1407 MB | 785 MB |

The latency win is for short commands. Long-form audio gains little (~1.04x on a
30s clip), because the cached prompt is a small share of that work — though the
memory saving grows with length.

Accuracy is unchanged: on LibriSpeech test-other (n=200) the two produce
byte-identical transcripts with no prompt (5.35% WER for both), and 5.33% vs
5.43% with a 50-name prompt.

## Docker Image (Vulkan: AMD and Intel GPUs)

This fork publishes one image, the Vulkan one below. The CPU image is
[`rhasspy/wyoming-whisper`](https://hub.docker.com/r/rhasspy/wyoming-whisper)
and the CUDA image is `Dockerfile.gpu`, both in the upstream repository
[OHF-Voice/wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper).


`Dockerfile.vulkan` runs Whisper on any GPU with a Mesa Vulkan driver — AMD
Radeon cards and **APU iGPUs that ROCm does not support** (e.g. the Radeon
Vega 7 in a Ryzen 5 5600G), and Intel graphics. It is published to the GitHub
Container Registry (amd64 only) by the `Docker (Vulkan)` workflow:

| Tag | Built from |
| --- | --- |
| `ghcr.io/frauhottelmann/wyoming-vulkan-whisper:latest`, `:<version>` | each GitHub release |
| `ghcr.io/frauhottelmann/wyoming-vulkan-whisper:main` | every push to `main` |

Every build is smoke-tested first: the container transcribes a test recording
through whisper.cpp's Vulkan backend (on Mesa's software Vulkan device, since
CI runners have no GPU) before anything is pushed. To build it yourself:

``` sh
git clone https://github.com/frauhottelmann/wyoming-faster-whisper.git
cd wyoming-faster-whisper
docker build -f Dockerfile.vulkan -t wyoming-whisper:vulkan .
```

CTranslate2 (and so faster-whisper) has no Vulkan backend, so this image
transcribes with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) through
the `whisper-cpp` backend instead, compiled with its Vulkan backend. Everything
else — the Wyoming protocol, Home Assistant discovery, name biasing, VAD
endpointing — is the same server. There is no ROCm in it: the image is about
550 MB of Python environment on top of Ubuntu, against several GB for a ROCm
stack.

The container needs the GPU's render node, and permission to open it:

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    ghcr.io/frauhottelmann/wyoming-vulkan-whisper:latest --language de
```

or with Compose:

``` yaml
services:
  whisper:
    image: ghcr.io/frauhottelmann/wyoming-vulkan-whisper:latest
    ports: ["10300:10300"]
    volumes: ["./data:/data"]
    devices: ["/dev/dri:/dev/dri"]
    # The numeric gid of the host's render group: getent group render
    group_add: ["993"]
    command: --language de
```

The image defaults to `--stt-library whisper-cpp` and `--device vulkan`
(`vulkan:1` picks the second GPU, `cpu` disables it). Check the log on the
first start — whisper.cpp reports what it found:

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV RENOIR) (radv) | uma: 1 | fp16: 1 | ...
whisper_backend_init_gpu: using Vulkan0 backend
```

If it says `ggml_vulkan: No devices found.` instead, the container cannot open
`/dev/dri/renderD128` (check `--device` and `--group-add`), and transcription
runs on the CPU.

Models are whisper.cpp's ggml files, named as whisper.cpp names them and
downloaded from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)
into `/data` on first use: `tiny`, `base`, `small`, `medium`, `large-v3`,
`large-v3-turbo`, each also as `.en` and as quantized `-q5_1`/`-q5_0`/`-q8_0`
variants. The default here is `small`. A path to a `.bin` file, or
`<owner>/<repo>/<file>.bin` for a ggml file in another HuggingFace repository,
works too:

``` sh
docker run ... ghcr.io/frauhottelmann/wyoming-vulkan-whisper:latest --model large-v3-turbo-q5_0 --language de
```

Notes:

- **iGPUs share system memory.** An APU's Vulkan heap is carved out of RAM; a
  `small` model needs well under 1 GB, but for `medium` and larger, raise the
  UMA frame buffer in the BIOS if the load fails with an out-of-memory error.
- **The faster-whisper names don't apply.** `--model small-int8` and the like
  name CTranslate2 conversions; use `small`, `small-q8_0`, etc.
- **`--vad-filter` is faster-whisper only.** `--vad-clip`, `--vad-endpointing`,
  `--initial-prompt`, `--beam-size`, `--whisper-task translate`, and
  `--hass-token` all work with `whisper-cpp`.
- **faster-whisper is still installed** and can be selected with
  `--stt-library faster-whisper`; it runs on the CPU. The other optional
  backends are left out of this image, since none of them can use Vulkan.
- The first transcription after a start is slower: the Vulkan driver compiles
  whisper.cpp's shaders then and caches them.

Outside Docker, `pip install 'wyoming-faster-whisper[whisper-cpp]'` installs
the CPU-only pywhispercpp wheel. For Vulkan, build it from source with
`GGML_VULKAN=1 pip install --no-binary pywhispercpp pywhispercpp` (needs
`glslc`, the Vulkan and SPIR-V headers, and a compiler). `Dockerfile.vulkan`
does the same from a git checkout, with a newer whisper.cpp than the one
pywhispercpp bundles.

Models are downloaded to `/data`, so make sure this points to a Docker volume.

### Running Without Internet Access

A model that is already downloaded is loaded from disk without contacting
Hugging Face, so a container that has been started once keeps working on a
network with no route out. Only a model that is *missing* is fetched.

That matters on a network which drops outbound traffic rather than refusing it —
a Docker bridge marked `internal`, for example. The update check that used to
run on every start would sit there until the TCP connection timed out, and with
`restart: always` the container boot-looped.

To rule out downloads entirely, pass `--local-files-only` (or
`WYO_WHISPER_LOCAL_FILES_ONLY=true`) once the models you want are in `/data`. A
model that is not there is then an error instead of a download:

```yaml
services:
  whisper:
    image: ghcr.io/frauhottelmann/wyoming-vulkan-whisper:latest
    volumes:
      - ./data:/data
    devices:
      - /dev/dri:/dev/dri
    group_add: ["993"]
    environment:
      WYO_WHISPER_MODEL: small
      WYO_WHISPER_LANGUAGE: en
      WYO_WHISPER_LOCAL_FILES_ONLY: "true"
    networks:
      - no-internet

networks:
  no-internet:
    internal: true
```

Both images point `HF_HOME`, `XDG_CACHE_HOME`, and `MODELSCOPE_CACHE` at `/data`
so the caches these libraries keep outside the model directory land on the
volume too. The largest is the Xet chunk cache used while downloading, which can
run to several GB. Outside Docker, set them yourself if you don't want them in
`~/.cache`.

### Health Check

The image carries a `HEALTHCHECK`, so `docker ps` reports `healthy` or
`unhealthy` and a Compose stack can wait on it with `depends_on:` /
`condition: service_healthy`. It sends the server a `Describe` and requires an
`Info` with an ASR program back, rather than only opening a socket: the port is
bound by the OS, so a connect-only check stays green even when the event loop is
wedged, while a round trip proves the accept loop and the event handler are both
still running.

The first 5 minutes don't count against it, because the server only starts
listening once the model is loaded — which means downloading it on first run —
and it takes 3 consecutive failures to turn the container unhealthy, because
transcription runs on the event loop and a check can time out behind a long
request.

To run it by hand:

``` sh
docker exec whisper \
    /usr/src/.venv/bin/python3 -m wyoming_faster_whisper.health_check
```

It prints nothing and exits 0 when healthy, or prints `unhealthy: <reason>` and
exits 1. That reason is what `docker inspect` shows under `.State.Health`.

The URI it checks is `tcp://127.0.0.1:10300`, or `WYO_WHISPER_URI` when the
container sets one (a listen-everywhere host like `0.0.0.0` is rewritten to
loopback). A `--uri` passed to `docker run` instead of the variable is invisible
to the check, so give it the same one:

```yaml
    healthcheck:
      test: ["CMD", "/usr/src/.venv/bin/python3", "-m",
             "wyoming_faster_whisper.health_check",
             "--uri", "tcp://127.0.0.1:10400"]
```

### GPU Without Docker

For a local install rather than a container, the two onnxruntime-based backends
have GPU counterparts of their extras — `onnx-asr-gpu` and `qwen3-asr-gpu`,
which pull `onnxruntime-gpu` in place of `onnxruntime`:

``` sh
pip install 'wyoming-faster-whisper[transformers,onnx-asr-gpu,qwen3-asr-gpu]'
```

faster-whisper additionally needs a CUDA-enabled CTranslate2 and torch built for
your CUDA version, and it depends on `onnxruntime` (for its bundled Silero VAD),
which will pull the CPU package back in and clobber `onnxruntime-gpu` — both
install the same `onnxruntime` module and whichever lands second wins, silently,
since the CPU provider still loads every model. Reinstall `onnxruntime-gpu`
last.

## Environment Variables

Every command-line option can also be set from the environment, which is what
Docker Compose gives you to configure a container without rewriting its
`command:`. The variable is the option's name, uppercased, with dashes as
underscores and a `WYO_WHISPER_` prefix:

| Option | Variable |
| --- | --- |
| `--model` | `WYO_WHISPER_MODEL` |
| `--language` | `WYO_WHISPER_LANGUAGE` |
| `--stt-library` | `WYO_WHISPER_STT_LIBRARY` |
| `--vad-endpointing` | `WYO_WHISPER_VAD_ENDPOINTING` |
| `--hass-token` | `WYO_WHISPER_HASS_TOKEN` |
| ...and so on for every option in `--help` | |

**Precedence**: a command-line argument always wins over the environment, so a
variable left over in a container can never silently override an explicit
argument. A variable that starts with `WYO_WHISPER_` but matches no option is
reported at startup rather than ignored, so a typo doesn't leave the server
running with a default you thought you had changed.

A few options don't take a plain string on the command line, so they read one
specially:

- **Flags** (`--debug`, `--vad-filter`, `--local-files-only`, ...) take
  `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`/empty. Unlike a plain "is it
  set?" test, `WYO_WHISPER_DEBUG=false` really does mean off.
- **`--data-dir`**, which can be repeated, splits on `:` —
  `WYO_WHISPER_DATA_DIR=/data:/media/models`. Passing any `--data-dir` on the
  command line replaces the list rather than adding to it.
- **`--vad-clip`**, which takes any number of values, splits on commas or
  spaces. Empty means the flag with no values, i.e. every library.
- **`--zeroconf`**, whose value is optional, takes the name to announce, or
  empty for the default name.

Any other variable left empty means the same as not setting it at all, which is
what a compose file or a `.env` produces for a value whose author left it blank.
An empty `WYO_WHISPER_URI` gets you argparse's "the following arguments are
required", not a server that starts up and fails on an empty string.

### Secrets

Each variable also has a `_FILE` form naming a file to read the value out of:
`WYO_WHISPER_HASS_TOKEN_FILE=/run/secrets/hass_token`. That is the convention
Docker Compose and Swarm secrets use — a secret is mounted as a file rather than
handed over as a variable — and it keeps a long-lived Home Assistant token out
of both the process's command line and its environment. The file is preferred
over the plain variable when both are set, and its trailing newline is stripped.

```yaml
services:
  whisper:
    image: ghcr.io/frauhottelmann/wyoming-vulkan-whisper:latest
    ports:
      - "10300:10300"
    volumes:
      - ./data:/data
    devices:
      - /dev/dri:/dev/dri
    group_add: ["993"]
    environment:
      WYO_WHISPER_MODEL: small
      WYO_WHISPER_LANGUAGE: en
      WYO_WHISPER_HASS_API: http://homeassistant.local:8123/api
      WYO_WHISPER_HASS_TOKEN_FILE: /run/secrets/hass_token
    secrets:
      - hass_token

secrets:
  hass_token:
    file: ./secrets/hass_token.txt
```

In the Docker image, `--uri`, `--data-dir`, and `--device` have defaults baked
into the entrypoint; those are dropped when the matching variable is set, so
`WYO_WHISPER_URI` and friends work there too.
