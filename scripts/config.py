import configparser
from tabulate import tabulate


def write_config(args):
    config = configparser.ConfigParser()
    config.add_section('config')

    config.set("config", 'corpus_name', args.corpus_name)
    config.set("config", 'images_folder', args.images_folder)
    config.set("config", 'part_corpus', str(args.part_corpus) if args.part_corpus else "")
    config.set("config", 'sift_step', str(args.sift_step))
    config.set("config", 'bin_sizes', ",".join([str(i) for i in args.bin_sizes]))
    config.set("config", 'magnitude_thresholds', ",".join([str(i) for i in args.magnitude_thresholds]))
    config.set("config", 'codebook_size', str(args.codebook_size))
    config.set("config", 'patch_height', str(args.patch_height))
    config.set("config", 'patch_widths', ",".join([str(i) for i in args.patch_widths]))
    config.set("config", 'patch_sampling', str(args.patch_sampling))
    config.set("config", 'topics', str(args.topics))

    # Write the new structure to the new file
    with open(r"corpus/" + args.corpus_name + "/config.ini", 'w') as configfile:
        config.write(configfile)

    config_str = []
    for k, v in dict(config["config"]).items():
        config_str.append(["[CONFIG]", k, "=", v])

    print(tabulate(config_str, tablefmt="plain"))


def read_config(corpus_name):
    config = configparser.ConfigParser()
    config.read(r"corpus/" + corpus_name + "/config.ini")

    return config["config"]
