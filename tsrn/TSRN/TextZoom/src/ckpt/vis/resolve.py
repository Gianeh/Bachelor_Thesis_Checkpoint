import torch

print("Loading trained model_best.pth ...")
old_state_dict = torch.load("./model_best.pth")

g_dict = old_state_dict["state_dict_G"]
new_g_dict = {f"module.{key}" : value for key,value in g_dict.items()}
old_state_dict["state_dict_G"] = new_g_dict

print("Performed conversion of state_dict_G !")
torch.save(old_state_dict, "./resolved_model_default_settings.pth")

print("Results saved as resolved_model_default_settings.pth")
