import random
import qai_hub

from qai_hub_models.models.ffnet_40s import Model

# High resolution variants
# from qai_hub_models.models.ffnet_40s import Model
# from qai_hub_models.models.ffnet_54s import Model
# from qai_hub_models.models.ffnet_78s import Model

from qai_hub_models.models.ffnet_78s_lowres import Model

# Low resolution variants
# from qai_hub_models.models.ffnet_78s_lowres import Model
from qai_hub_models.models.ffnet_122ns_lowres import Model

from torchinfo import summary
from utils import get_ai_hub_api_token


## ------------------------------------------------------ ##
model = Model.from_pretrained()
input_shape = (1, 3, 1024, 2048)
stats = summary(model,
  input_size=input_shape,
  col_names=["num_params", "mult_adds"]
)
print(stats)

## ------------------------------------------------------ ##
model = Model.from_pretrained()
print(f"Pretrained model is: {model.__class__.__name__}\n")
low_res_input_shape = (1, 3, 512, 1024)
stats = summary(model,
                  input_size = low_res_input_shape,
                  col_names = ["num_params", "mult_adds"]
                )
print(stats)

## ------------------------------------------------------ ##
low_res_input_shape = (1, 3, 512, 1024)

model = Model.from_pretrained()
print(f"Pretrained model is: {model.__class__.__name__}\n")
hi_res_models = ["FFNet40S", "FFNet54S", "FFNet78S"]
low_res_models = ["FFNet78SLowRes", "FFNet122SLowRes"]

if model in hi_res_models:
    stats = summary(model,
                      input_size = input_shape,
                      col_names = ["num_params", "mult_adds"]
                    )

if model in low_res_models:
    stats = summary(model,
                      input_size = low_res_input_shape,
                      col_names = ["num_params", "mult_adds"]
                    )

print(stats)
## ------------------------------------------------------ ##
ai_hub_api_token = get_ai_hub_api_token()

## ------------------------------------------------------ ##
devices = [
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy Tab S8",
    "Xiaomi 12",
    "Xiaomi 12 Pro",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
]

selected_device = random.choice(devices)
print(selected_device)
