# Notice
fix windows not support torchaudio_extension sox

find this codes where is lib\site-packages\modelscope\pipelines\audio\speaker_verification_light_pipeline.py 
 class name SpeakerVerificationPipeline  function: preprocess

```python
     data, fs = torchaudio.sox_effects.apply_effects_tensor(
                        data,
                        fs,
                        effects=[[
                            'rate',
                           str(self.model_config['sample_rate'])
                        ]])
```

update to 

```python
resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.model_config['sample_rate'])
data = resampler(data)
```