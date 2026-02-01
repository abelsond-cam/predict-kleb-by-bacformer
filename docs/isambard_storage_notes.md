# Isambard storage and where to save filtered ESMC data

## From Isambard documentation

**Source:** [Storage spaces](https://docs.isambard.ac.uk/user-documentation/information/system-storage), [File transfer – public shared storage](https://docs.isambard.ac.uk/user-documentation/guides/file_transfer/#using-public-shared-storage).

### Project public storage (`$PROJECTDIR_PUBLIC`)

- **Path:** `/projects/public/<PROJECT>` → for project `u5ah`: **`/projects/public/u5ah`**
- **Permissions:** Writable by **members of the owning project only**; readable by all users on the facility.
- **Use:** Sharing data with other projects; good place for large outputs that need to stay on the same filesystem as the source data.

So **if you are a member of project `u5ah`**, you can write to `/projects/public/u5ah/`, including subdirectories such as `/projects/public/u5ah/data/` (or a dedicated folder for filtered ESMC data). The raw parquet data is already under `/projects/public/u5ah/data/genomes/atb/esmc-large`, so writing filtered data under `/projects/public/u5ah/data/` keeps everything on the same project public space.

### Quotas (why not to use $HOME for large outputs)

| Storage           | Path (example)              | Capacity (typical) | Notes                          |
|------------------|-----------------------------|--------------------|--------------------------------|
| **$HOME**        | `/home/<PROJECT>/<USER>...`  | **100 GiB**        | Not for large data; small quota |
| **$PROJECTDIR**  | `/projects/<PROJECT>`       | 20–200 TiB         | Project-only access             |
| **$PROJECTDIR_PUBLIC** | `/projects/public/<PROJECT>` | (project quota)   | Same project space, world-readable |
| **$SCRATCHDIR**  | `/scratch/<PROJECT>/...`     | 5 TiB              | Isambard 3: purge if not accessed for 60 days |

Filtered Klebsiella data at **>50 GB** would quickly use half of a 100 GiB $HOME; it should go on **project storage** (e.g. under `/projects/public/u5ah/`) instead.

### Checking quota (on Isambard)

```bash
# Project public directory (replace with your actual path if different)
lfs quota -hp $(lfs project -d /projects/public/u5ah | awk '{print $1}') /projects/public/u5ah
```

---

## Recommended location for filtered parquets

- **Preferred:** e.g. **`/projects/public/u5ah/data/esmc_embeddings`** (or another subdir under `/projects/public/u5ah/data/`) so filtered data lives next to the raw data and uses project quota, not $HOME.
- **Avoid:** Saving tens of GB under `$HOME` or under your workspace if that is on $HOME.

When we implement the filter script, we can make the default `--output-dir` configurable so you can point it to `/projects/public/u5ah/data/esmc_embeddings` when running on Isambard.

---

## Testing write access on Isambard

Run these **on an Isambard login node** (where `/projects/public/u5ah` is mounted) to confirm you can write under project public storage.

**1. Check if the directory exists and list it**
```bash
ls -la /projects/public/u5ah/data/
```

**2. Test writing a small file**
```bash
echo "write test $(date)" > /projects/public/u5ah/data/.write_test_$$.txt && cat /projects/public/u5ah/data/.write_test_$$.txt
```
If that succeeds, you can write to `/projects/public/u5ah/data/`. Remove the test file:
```bash
rm -f /projects/public/u5ah/data/.write_test_$$.txt
```

**3. (Optional) Test creating the intended output subdir**
```bash
mkdir -p /projects/public/u5ah/data/klebsiella_esmc_embeddings
touch /projects/public/u5ah/data/esmc_embeddings/.write_ok
rm /projects/public/u5ah/data/esmc_embeddings/.write_ok
```
If all steps succeed, you can safely use `/projects/public/u5ah/data/esmc_embeddings` as the output directory for the filtered parquet script.
