# %%
# ensure PIP is updated
# %pip install -U pip

# %%

%pip install stability-sdk

# %%
import getpass, os

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = getpass.getpass('Enter your API Key')

# %%

import io
import os
import warnings

from IPython.display import display
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# connect
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-768-v2-0",
)

# load mask image from local files
gradient_mask = Image.open('w_to_b_mid.png')

# %%

seed = 1212
prompt = "isometric behance octane render highly detailed"

# %%

# generate a test image
answers = stability_api.generate(
    prompt=prompt + " cyan",
    start_schedule=1,
    seed=seed,
    steps=30,
    cfg_scale=7.0,
    width=768,
    height=768,
    sampler=generation.SAMPLER_K_DPMPP_2M
)

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn("Banana alert!")
        if artifact.type == generation.ARTIFACT_IMAGE:
            seed = artifact.seed
            print(seed)
            img = Image.open(io.BytesIO(artifact.binary))
            display(img)

#%%

# set up transformation matrix
a, b, d, e = 1, 0, 0, 1
c = 0  # left / right
f = 256  # up / down

source = img
output = [source]

# optionally vary the prompt with each iteration
prompts=[
    prompt + " magenta",
    prompt + " yellow",
    prompt + " cyan ants",
    prompt + " magenta",
    prompt + " yellow",
    prompt + " cyan birds",
    prompt + " magenta",
    prompt + " yellow",
    prompt + " cyan locusts",
    ]

# custom count or use prompts array length
# count = 8
count = len(prompts)

# iterate `count` times
for i in range(count):

  answers = stability_api.generate(
      prompt=prompts[i],
      init_image=source,
      mask_image=gradient_mask,
      start_schedule=1,
      seed=seed + i,
      steps=30,
      cfg_scale=7.0,
      width=768,
      height=768,
      sampler=generation.SAMPLER_K_DPMPP_2M 
)

  for resp in answers:
      for artifact in resp.artifacts:
          if artifact.finish_reason == generation.FILTER:
              warnings.warn("Boobies alert!")
              break
          if artifact.type == generation.ARTIFACT_IMAGE:
              result = Image.open(io.BytesIO(artifact.binary))
              output.append(result)
              thumb = result.resize((round(result.width * .25), round(result.height * .24)))
              display(thumb)
              # now translate output for next round
              source = result.transform(result.size, Image.AFFINE, (a, b, c, d, e, f))

# %%

# create collage image

output_count = len(output)
collage = Image.new('RGB', (768, 256 + output_count * 256))

for i in range(1, output_count):
  collage.paste(im=output[i], box=(0, (i-1) * 256))

display(collage)

# %%

# save to a file

collage.save('collage.png')

# %%

# generate a 768x432 scrolling video from the collage
height = (1 + output_count) * 256
height = height - 432
rate = height / 32

! rm scroll.mp4
! unsetopt nomatch; ffmpeg -r 1 -loop 1 -t 33 -i collage.png -filter_complex "color=white:s=768x432, fps=fps=30[bg];[bg][0]overlay=y=-{height}+'t*{rate}':shortest=1[video]" -preset ultrafast -map [video] scroll.mp4
