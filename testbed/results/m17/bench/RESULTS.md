# M17.3 S4 cost attribution: the record append (2026-09-26, WSL, project on /mnt/c = 9P)

`bench_record.py` (run in ~/venvs/ci, cwd ~/ai_compiler):

| what | ms/call |
|---|---|
| append on /mnt/c (project folder), open/append/close per line | 4.633 |
| append on ext4 (/tmp), open/append/close per line | 0.005 |
| cache_key_contract.check PASS path, record on ext4 | 0.021 |
| load.safely(no-op), record on ext4 | 0.008 |
| load.safely(no-op), record on /mnt/c | 4.387 |

`bench_append.py` (same venv; variants on /mnt/c):

| variant | ms/call |
|---|---|
| open/append/close per line | 2.462 |
| kept open, write + flush per line | 0.176 |
| kept open, buffered write (flush at close) | 0.004 |
| kept fd, os.write per line | 0.186 |

The record.py change keeps files open with a flush per line (the second variant).
