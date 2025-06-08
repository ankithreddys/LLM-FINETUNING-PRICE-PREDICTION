from huggingface_hub import login

def login_to_huggingface(hf_token):
    login(hf_token, add_to_git_credential=True)
    print("=====  Authentication Successfull  ====")