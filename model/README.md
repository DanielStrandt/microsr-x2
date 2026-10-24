# Model directory

The repository includes the V6 model as **`microsr_v6.msr`** (495 bytes).

Model weights are distributed separately from the MIT-licensed software. Read `LICENSE_NOTICE.md` for the model terms. The synthetic unit-test models are not trained for useful super-resolution.

From the repository root (assuming trainer exports are in the parent directory):

```powershell
python microsr_model.py pack --export "../microsr_int2" --output "./model/microsr_v6.msr"
python microsr_model.py info --model "./model/microsr_v6.msr"
```

The `.gitignore` file deliberately **does not ignore `.msr`** so the published model remains available with the source.

Do **not** commit the training dataset or large `.pt` training checkpoints to this lightweight inference repository. The `.pt` file is not required for deployment.
