def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# function to return key for any value in a dict
def get_key(my_dict, val): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key