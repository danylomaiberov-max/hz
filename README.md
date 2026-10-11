# ollama-xp — локальный запуск LLM на 32-битной Windows XP

Однофайловая программа `ollama.exe` (PE32, i386, без внешних DLL, подсистема 5.1), которая повторяет
интерфейс и функции [Ollama](https://ollama.com): те же команды, тот же REST API на порту `11434`,
тот же формат `Modelfile`. Вычисления — через [llama.cpp](https://github.com/ggml-org/llama.cpp) (только CPU).

## Что перенесено из Ollama

| Область | Что есть |
|---|---|
| CLI | `serve`, `run`, `pull`, `list`/`ls`, `ps`, `show` (`--modelfile --parameters --template --system --license`), `create -f`, `cp`, `rm`, `stop`, `--version` |
| Интерактивный чат (`run`) | приглашение `>>>`, многострочный ввод `"""`, `/set parameter\|system\|verbose\|quiet\|format`, `/show`, `/save`, `/load`, `/clear`, `/bye`, `--verbose` со статистикой, ввод через pipe, Ctrl+C прерывает генерацию |
| Modelfile | `FROM` (`.gguf` или имя модели), `PARAMETER` (в т.ч. несколько `stop`), `SYSTEM`, `TEMPLATE` (Go-шаблоны), `MESSAGE`, `LICENSE` |
| Шаблоны | подмножество Go `text/template`: `.System .Prompt .Response`, `range .Messages`, `if/else if/else`, `eq ne not and or`, `{{- -}}`; если `TEMPLATE` нет — берётся шаблон из GGUF, иначе ChatML |
| REST API | `GET /`, `/api/version`, `/api/tags`, `/api/ps`; `POST /api/generate`, `/api/chat` (потоковый NDJSON), `/api/show`, `/api/embed`, `/api/embeddings`, `/api/pull`, `/api/create`, `/api/copy`, `DELETE /api/delete`; предзагрузка/выгрузка через `keep_alive` |
| OpenAI-совместимость | `/v1/chat/completions`, `/v1/completions` (в т.ч. SSE), `/v1/models`, `/v1/embeddings` |
| Хранилище | `%USERPROFILE%\.ollama\models` (`OLLAMA_MODELS`), блобы по SHA-256 + манифесты по тегам, дедупликация, `name:tag`, `latest` по умолчанию |
| Управление памятью | одна загруженная модель, `OLLAMA_KEEP_ALIVE` (по умолчанию 5 минут), повторное использование KV-кэша между репликами чата |
| Переменные | `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_THREADS`, `OLLAMA_DEBUG`, `OLLAMA_MIRROR` |

## Сборка (на Linux, кросс-компиляция)

```sh
apt install cmake g++-mingw-w64-i686-posix
scripts/build-xp.sh            # скачает llama.cpp нужной ревизии, применит XP-патч и соберёт build-xp/ollama.exe
scripts/build-xp.sh -DXPLLM_SSE2=OFF   # для процессоров без SSE2 (до Pentium 4 / Athlon 64)
```

Сделано для совместимости с XP:

* `third_party/llama-winxp.patch` — заменяет в ggml вызовы Vista+ (`SRWLOCK`, `CONDITION_VARIABLE`, `InitOnceExecuteOnce`)
  на `CRITICAL_SECTION` и простые аналоги;
* `src/xp_compat.c` — подменяет `GetTickCount64`, который статически подтягивает `libwinpthread`;
* `-D_WIN32_WINNT=0x0501`, версия ОС/подсистемы PE = 5.1, `--large-address-aware` (до 3 ГБ адресного пространства при `/3GB`), статическая линковка;
* ввод/вывод консоли через `ReadConsoleW`/`WriteConsoleW` (Unicode), сокеты через Winsock 2 (`ws2_32`, есть в XP).

Проверка: из таблицы импорта `ollama.exe` исключены все функции новее XP (проверено `objdump -p`); программа
запускалась в Wine с режимом `winxp` — `pull`, `list`, `show`, `run`, `serve`, `/api/*`, эмбеддинги работали на реальной
GGUF-модели. **На физической/виртуальной Windows XP сборка не запускалась** — проверьте у себя.

Без `third_party/llama.cpp` проект собирается с заглушкой-движком (`ollama --version` покажет `mock engine`) — удобно для
отладки CLI/API на любой ОС: `cmake -S . -B build && cmake --build build`. Тесты ядра: `-DXPLLM_TESTS=ON`, `./build/test_core`.

## Использование

```bat
:: Модель берётся из локального файла или по http://
ollama pull tiny C:\models\stories15M-q4_0.gguf
ollama pull qwen http://192.168.0.5:8000/qwen2.5-0.5b-instruct-q4_k_m.gguf

ollama list
ollama run qwen "Привет!"
ollama run qwen                      :: интерактивный чат
ollama serve                         :: REST API на 127.0.0.1:11434
curl http://127.0.0.1:11434/api/chat -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
```

`Modelfile`:

```
FROM C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf
PARAMETER temperature 0.4
PARAMETER num_ctx 1024
SYSTEM Ты — краткий помощник.
```
`ollama create helper -f Modelfile`

Чтобы `ollama pull qwen` работал без аргументов, добавьте строку `qwen:latest http://хост/файл.gguf` в `catalog.txt`
(рядом с каталогом `models`) или задайте `OLLAMA_MIRROR=http://хост/{name}-{tag}.gguf`.

## Ограничения (почему не 1:1 с Ollama)

* **Нет HTTPS.** В XP нет современного TLS, поэтому `pull` из `registry.ollama.ai` / Hugging Face напрямую невозможен.
  Скачайте `.gguf` на другом компьютере и отдайте по `http://` (например, `python -m http.server`) либо скопируйте файл.
  `push` не поддерживается.
* **32 бита = ~2 ГБ адресного пространства** (до 3–4 ГБ с `/3GB` + LAA). Практически — модели до ~1–3B параметров в Q4/Q5
  и контекст 512–2048. Файлы читаются через mmap, но веса + KV-кэш должны поместиться в адресное пространство.
* Только CPU, без GPU. Скорость на старых процессорах низкая; число потоков — `OLLAMA_NUM_THREADS`.
* Не реализовано: изображения (vision), tool calling, thinking, структурированный `format` (JSON-схема; режим `--format json`
  работает лишь подсказкой в промпте), LoRA `ADAPTER`, квантизация при `create -q`, загрузка блобов `/api/blobs`.
* Консоль XP не умеет выводить многие Unicode-символы шрифтом по умолчанию — для кириллицы выберите шрифт Lucida Console в свойствах окна.
* Многопоточность llama.cpp в XP-патче использует опрос вместо condition variable (небольшие накладные расходы при простое потоков).
