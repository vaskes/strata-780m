# Strata на Radeon 780M (gfx1103) — llmhost2, 2026-10-20

## Коротко

Strata поднята на iGPU 8700G из исходников, в Docker, и **генерирует токены**.

**Итог по генерации: 9.4 tok/s — лучшее из замеров, десятку не отжали.**
Декод 6.2–9.4 tok/s, приём MTP-драфта 76–93%. Потолок — SSD, не вычисления:
~16 ГиБ экспертов физически не влезают в 58 ГиБ RAM и стримятся с диска
(66 секунд ожидания диска в одном запросе).

Модель качать не пришлось: готовая лежала на хосте. Long-prefill упёрся в
**rocBLAS**, который на gfx1103 сломан — подтверждено независимым пробником,
без Strata вовсе.

---

## 1. ROCm или Vulkan — вопрос закрыт

**Vulkan в Strata нет вообще.** Ноль совпадений `vulkan` / `GGML_VULKAN` в
`CMakeLists.txt` (152 КБ) и во всём дереве. Бэкенды ровно три: `STRATA_ENABLE_CUDA`,
`STRATA_ENABLE_HIP`, `STRATA_ENABLE_SYCL`. Для 780M — только HIP (ROCm).

Про gfx1103 у проекта есть отдельная страница `docs/RADEON_780M.md`: статус
`experimental and opt-in`, замеры на одной машине (Ryzen 7 8745HS), готовых
движков под gfx1103 **не существует** — ни в релизах, ни в ghcr
(`hip-rx7900` собран под gfx1100/1101/1102 и на 780M не стартует).

## 2. Модель — ничего не качали

На llmhost2 уже был `Qwen3.8-Flash-Next` **UD-IQ4_XS** (3 шара, 93.7 ГБ):
`/opt/models/Qwen3.8-Flash-Next-GGUF/`. SHA256 **всех трёх** совпали с пином
в `setup.py` байт-в-байт:

```
5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4  -00001-of-00003
577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7  -00002-of-00003
d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833  -00003-of-00003
```

Это ровно та модель, которую мерили в `RADEON_780M.md` и в `STRIX_HALO.md`.

Память: у UD-IQ4_XS 59.5 ГиБ экспертов. `setup.py:unsloth_budget()` для UMA-APU
сам форсирует `--resident-budget-gib` (#1715) — остальное читается с SSD.

## 3. Что собрано и запущено

- Хост: `/opt/strata-780m/` (движок, логи, скрипты, бандл)
- Образ: `strata:0.2.1-rocm-gfx1103` (1.37 ГБ)
- Пакет движка: `0.1.46.1-linux-x86_64-hip-gfx1103`, 274 файла, 608 МБ
  - `BUILD.json: {"archs":"gfx1103","hip":"10.1","hipblaslt_version":100401}`
  - `tuning/gfx1103-hipblaslt-100401.txt` — калиброванная таблица внутри пакета
  - `rocblas/library/` 150 gfx1103-ядер, `hipblaslt/library/gfx1103/` 98 файлов
- Сервис: `docker run` c `/dev/kfd`, `/dev/dri`, 32K ctx, KV int8
- Движок самodetect'нул карту и сам включил матриц-корный набор:
  `integrated GPU (gfx1103): the matrix-core prompt set is on (HIP_WMMA PF_GEMM PA_FAST SELECT_WMMA HC_UPMIX)`

### ROCm: хостовый не годится, в Docker — да

| ROCm | hipBLASLt | Вердикт |
|---|---|---|
| `/opt/rocm` хоста, 7.2.4 | **1.2.0** | нет таблицы gfx1103 (нужна 1.4.1 / 100401) |
| `rocm/dev-ubuntu-24.04:10.1.0-full` | **1.4.1 (100401)** | ✅ та, на которую откалибрована gfx1103-таблица |

Хостовый ROCm остался нетронутым — ядра в контейнере.

## 4. Замеры (реальные, из лога движка)

`strata serve: prompt N tokens = ... read in X ms (Y tok/s), G generated in Z ms (W tok/s)`

### Генерация (decode) — итог

| промпт | сгенерировано | decode | приём драфта |
|---|---|---|---|
| 64 tok | 32 | **9.4 tok/s** (3387 мс) | 16/21 (76%) |
| 64 tok | 32 | **9.4 tok/s** (3389 мс) | 16/19 (84%) |
| 55 tok | 29 | 9.0 tok/s (3238 мс) | 13/14 (93%) |
| 55 tok | 29 | 8.9 / 8.8 / 7.8 tok/s | 13/14 |
| 53 tok | 1 | 5.7–6.6 tok/s | — (окно не набирается) |

**Лучшее: 9.4 tok/s. Десятку не отжали.**

Одиночные токены (5.7–6.6) — это не чистый декод: MTP-драфтеру не из чего набирать
окно, поэтому такая скорость означает «стоимость одного шага», а не поток.

Потолок здесь упирается в SSD, а не в вычисления: в логе движка один запрос показал
`decode waited 66362.0 ms in 16622 fetches` — 66 секунд чистого ожидания диска.
Причина structural: у UD-IQ4_XS 59.5 ГиБ экспертов, в коробке 58 ГиБ RAM, то есть
~16 Гиб экспертов физически не могут быть резидентными. CPU-тир ужат до ~0.1–5.5 ГиБ.

### Префилл — только короткие промпты

| промпт | prefill |
|---|---|
| 55 tok | **23.8 tok/s** (2315 мс) |
| 64 tok | 21.3 / 15.2 tok/s |
| 53 tok | 9.5–17.6 tok/s |

Выше ~85 токенов prefill не проходит вовсе (см. §5). Так что честное число
«23.8 tok/s» относится к 55 токенам, а не к рабочему диапазону.

- Кэш экспертов при декоде: **89–96%** попаданий.
- `GPUActive: 53.1 GiB` — эксперты живут в unified memory, как и задумано.
- GTT = 51 GiB (`ttm.pages_limit`), `cwsr_enable=0` — ровно как советует
  `AMD_HIP.md` для 780M.

**Сравнение с их замерами** (8745HS, 60 ГБ, TheRock SDK): у них декод 13–16 tok/s
на модели вдвое меньшей арены (Q2_0, 34 ГиБ против наших 59.5 ГиБ) — она целиком
влезает в RAM, ours стримит с SSD. Это и есть разница 9 против 14.

## 5. Блокер: rocBLAS на gfx1103 сломан

Длинные промпты падают: `prefill gemm: cublasGemmEx: cuBLAS status 6`.

Я это сузил независимым пробником, **в чистом `rocm/dev` контейнере, без Strata**:

```c
rocblas_sgemm(512, 512, 512)      -> OK          // проходит
rocblas_sgemm(512, 2560, 2560)    -> Memory access fault by GPU node-1
                                    Reason: Page not present or supervisor privilege
```

2560 — это ширина dense-слоя модели, ровно та форма, которую гоняет prefill.

Причина видна в самом ROCm: **gfx1103 нет в списке arch у rocBLAS**.

- ROCm 7.2.4 перечисляет `TensileLibrary_lazy_` для gfx1102, gfx1150, gfx1151,
  gfx1200, gfx1201, gfx950, gfx90a, gfx908, gfx1030 — **gfx1103 отсутствует**.
- ROCm 10.1 имеет только `*_fallback_gfx1103.*` верхнего уровня без поддиректория
  `gfx1103/` — и именно они падают.

Упаковка не виновата: те же GEMM из **упакованного** `librocblas.so.5` внутри
компонента Strata ведут себя так же, `ldd` чист, все 98 hipBLASLt-файлов на месте.
Отключение WMMA-набора и hipBLASLt-таблицы картину не меняет.

**Как чинить:** собирать не официальным ROCm-образом, а **TheRock gfx110X-all SDK**
(`rocm.nightlies.amd.com/v2/gfx110X-all/`) — ровно то, что делает сам проект:
`setup.py:ROCM_INDEXES["gfx1103"]` и `packaging/build_hip_gfx1103.sh`
(пин `gfx110X-all → 7.10.0a20251121`). У TheRock настоящие gfx1103-ядра.

## 6. Грабли, на которые уже наступил (чтобы не переоткрывать)

1. **`STRATA_EXPERIMENTAL_GFX1103=1` обязателен.** Без него `setup.py` выкидывает
   gfx1103 из `AMD_ARCHS`, считает карту неподдерживаемой и **отказывается писать
   run-config** для любого семейства (`launch` exit 5). Дальше opt-in сам пишет все
   17 exact-свитчей в `cfg["env"]` — руками передавать не нужно.
2. **`--hipblaslt-tuning` не существует.** Таблица лежит в пакете и подхватывается
   по `BUILD.json:hipblaslt_version`. Флаг → движок печатает help и выходит.
3. **`collect_libs.py`: `--data-arch gfx1103`, а НЕ `--data-glob '*gfx1103*'`.**
   Glob молча вырезает файлы hipblaslt без arch в имени. Дальше — `hipErrorInvalidKernelFile`.
4. **`package_engine.py` требует `--tree`**, иначе в пакет не попадает дерево
   `rocblas/library` и движок умирает на старте с `Could not initialize Tensile host`.
5. **`BUILD.json` без `hipblaslt_version` — отказ упаковщика.** Заполняется штатным
   `packaging/stage_tuning.py --archs gfx1103 --rocm /opt/rocm --expect gfx1103`.
6. **Сеть рвёт соединение ровно на 62 914 560 байт (60 МиБ).** Убило и 66-МБ
   cpython-сборку, и llama.cpp-зип. Лечится `curl -r <offset>-<end>` по 30 МиБ
   + склейка; пиновый SHA256 после этого сходится.
7. **`set -o pipefail` + `ls | head -5` убивает скрипт** по SIGPIPE.
8. `--batch` конфликтует с `parallel`-override из entrypoint.
9. **Связка песочница↔хост идёт ~28 КБ/с**, а хост↔хост по mesh — **100 МБ/с
   (315 МБ за 2.9 с)**. Любой большой файл на llmhost2 гнал через llmhost1.
10. GitHub с хостов в этой сессии проседал до ~40 КБ/с, TLS-обмен ловил
    `server certificate verification failed`. Локальный бандл + mesh надёжнее.

## 7. Что на хосте

```
/opt/strata-780m/
  strata/                      исходники Strata @ main + llama.cpp 3cf03257
  build-780m/strata           движок, 35.5 МБ
  engine-pkg/0.1.46.1-...-hip-gfx1103/   пакет компонента, 608 МБ
  build-engine-780m.sh        сборка движка (воспроизводима)
  run-strata-780m.sh          запуск сервиса
  bench-780m.py / bench-find-ceiling.py / gemm-shapes-probe.cpp / rocblas-probe.c
docker image strata:0.2.1-rocm-gfx1103
volume strata-780m-data       /data (модель уже собрана, повторный старт быстрый)
```

Модель и хостовый ROCm 7.2.4 не тронуты. `/opt` свободен 1.1 ТБ.

## 8. Следующий шаг (одна команда conceptually)

Пересобрать движок, подменив ROCm-образ на TheRock gfx110X-all SDK 7.14/7.10 —
и prefill должен заработать; decode при этом, по их данным, не пострадает
(WMMA-набор даёт 1.6–2× на prompt, на decode 0.999).