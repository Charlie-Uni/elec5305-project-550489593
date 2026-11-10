"""Run the unmodified course STFT in MATLAB, then compare aligned complex bins.

The teaching source is supplied locally and is not redistributed by this project.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from esc_robustness.artifacts import sha256_file, write_json
from esc_robustness.data import load_audio
from esc_robustness.dsp import FeatureConfig, periodic_hann, stft


def validate(source, matlab, audio_file, output):
    source, matlab, audio_file, output = map(Path, (source, matlab, audio_file, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    # MyStft.m declares function stft; MATLAB requires a matching file name.
    shutil.copyfile(source, output / "stft.m")
    config = FeatureConfig()
    impulse = np.zeros(31)
    impulse[15] = 1.0
    cases = {
        "short_impulse": impulse,
        "silence": np.zeros(2048),
        "random_nonhop": np.random.default_rng(5305).normal(size=10007),
        "sine": np.sin(2 * np.pi * 17 * np.arange(8192) / 1024),
        "esc50_audio": load_audio(audio_file, config),
    }
    savemat(output / "inputs.mat", {"cases": cases, "window": periodic_hann(1024)})
    quoted = str(output.resolve()).replace("'", "''")
    script = f"""taskroot = '{quoted}';
addpath(taskroot, '-begin');
assert(strcmp(which('stft'), fullfile(taskroot, 'stft.m')));
load(fullfile(taskroot, 'inputs.mat'));
case_names = fieldnames(cases); reference = struct();
for k=1:numel(case_names)
    key=case_names{{k}};
    [value, original_length] = stft(cases.(key), 512, window(:), 0);
    reference.(key)=value;
end
matlab_version=version; resolved_function=which('stft');
save(fullfile(taskroot,'reference.mat'),'reference','matlab_version','resolved_function','-v7');
disp('COURSE_STFT_REFERENCE_COMPLETE');
"""
    (output / "run_reference.m").write_text(script)
    with (output / "matlab.log").open("w") as log:
        subprocess.run(
            [str(matlab), "-batch", f"run('{quoted}/run_reference.m')"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reference = loadmat(output / "reference.mat", simplify_cells=True)
    records = []
    for name, audio in cases.items():
        actual = stft(audio, config)
        expected = reference["reference"][name][:513, 2 : 2 + len(actual)].T
        error = float(np.max(np.abs(actual - expected)))
        relative = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(actual), 1e-30))
        passed = bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))
        records.append(
            dict(
                case=name,
                frames=len(actual),
                max_abs_error=error,
                relative_l2_error=relative,
                passed=passed,
            )
        )
    result = {
        "passed": all(r["passed"] for r in records),
        "matlab_version": reference["matlab_version"],
        "course_source_sha256": sha256_file(source),
        "dsp_sha256": sha256_file(Path(__import__("esc_robustness.dsp", fromlist=["x"]).__file__)),
        "audio_sha256": sha256_file(audio_file),
        "inputs_sha256": sha256_file(output / "inputs.mat"),
        "reference_sha256": sha256_file(output / "reference.mat"),
        "cases": records,
        "conventions": {
            "window": "Explicit identical periodic Hann, length 1024; hop 512; no FFT padding",
            "course_padding": "1024 leading zeros and additional trailing zeros",
            "comparison": "Course columns 3 onward, first 513 bins, transposed to frames x bins",
            "scope": "Matching waveform frames and complex FFT values; boundary policies differ",
            "source": "Unmodified teaching code copied under its declared function name stft.m",
            "default_window": "Course window helper is not used; window conventions are aligned explicitly",
        },
    }
    write_json(output / "validation.json", result)
    if not result["passed"]:
        raise AssertionError("Course STFT numerical comparison failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "matlab", "audio", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.matlab, args.audio, args.output), indent=2))
