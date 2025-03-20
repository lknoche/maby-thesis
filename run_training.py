import logging

from train import main

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)

server_info = {
    "host": "staffa.bio.ed.ac.uk",
    "username": "upload",
    "password": "gothamc1ty",
}


main(
    base_dir="/Users/pswain/wip/aliby_output/maby/",
    gfp_type="Hog1_GFP",
    num_epochs=3,
    server_info=server_info,
)
