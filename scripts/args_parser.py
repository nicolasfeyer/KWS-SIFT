import sys
import argparse


# Function to write un sys.stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        eprint(message + "\n")
        # get all helps from subparser
        # retrieve subparsers from parser
        subparsers_actions = []

        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_actions.append(action)

        if subparsers_actions:
            for subparsers_action in subparsers_actions:
                # get all subparsers
                for choice, subparser in subparsers_action.choices.items():
                    # subparsers_helps[choice] = subparser.format_help()
                    eprint("Action '{}'".format(choice))
                    eprint(subparser.format_help())
        else:
            self.print_usage(sys.stderr)

        sys.exit(1)


def parse_args():
    # ...
    parser = MyParser(
        prog="SIFT-KWS"
    )

    subparsers = parser.add_subparsers(help="Available actions", dest="action", required=True)

    generate_parser = subparsers.add_parser("generate", description="Generate the corpus", help="Generate the corpus")
    query_parser = subparsers.add_parser("query", description="Query the corpus", help="Query the corpus")

    for sub_p in [generate_parser, query_parser]:
        sub_p.add_argument("-cn", "--corpus-name", required=True, type=str,
                           help="""The name of the corpus to generate/to be used.""")

    ##############################
    # GENERATE PARSER
    ##############################

    # path of the image folders
    generate_parser.add_argument("-i", "--images-folder", required=True, type=str,
                                 help="""The path of the image folders.""")

    generate_parser.add_argument("-pt", "--part-corpus", required=False, type=str,
                                 help="""Images index to be used to build the corpus. By default, all the image in the 'images-folder' are used. First index is 0""")

    generate_parser.add_argument("-ss", "--sift-step", required=True, type=int,
                                 help="""The SIFT sampling step, i.e. the distance between SIFT descriptors during their extraction""")

    generate_parser.add_argument("-bs", "--bin-sizes", required=True, nargs="+", type=int,
                                 help="""The sizes of one of the 16 spatial bin of the SIFT descriptor in pixels""")

    generate_parser.add_argument("-mt", "--magnitude-thresholds", required=True, nargs="+", type=float,
                                 help="The magnitude limit below which a SIFT descriptor is discarded")

    generate_parser.add_argument("-k", "--codebook-size", required=True, type=int,
                                 help="""Size of the codebook, i.e. the number of clusters in which the SIFT descriptors are grouped""")

    generate_parser.add_argument("-H", "--patch-height", required=True, type=int,
                                 help="""The height of the patches""")

    generate_parser.add_argument("-ws", "--patch-widths", required=True, nargs="+", type=int,
                                 help="The widths of the patches")

    generate_parser.add_argument("-ps", "--patch-sampling", required=True, type=int,
                                 help="The sampling step of the patch")

    generate_parser.add_argument("-t", "--topics", required=True, type=int,
                                 help="""The number of topics used for the LSA transformation. It must be equal or lower than the codebook size K""")

    ##############################
    # QUERY PARSER
    ##############################

    query_parser.add_argument("-te", "--templates-folder", required=True, type=str,
                              help="""The path of the templates folder.""")

    # only-from-corpus : only the templates from the corpus part are used for the querying
    # only-from-non-corpus : only the templates from the non-corpus part are used for the querying
    # intersection : only the word from the non-corpus parts that are in the corpus part
    # union : the union all the templates are used for the querying
    query_parser.add_argument('-st', '--strategy', default='union', const='union', nargs='?', required=False,
                              choices=['only-from-corpus', 'only-from-non-corpus', 'intersection', "union"],
                              help='''The strategy to use for the querying.''')

    query_parser.add_argument("-tt", '--template-text-limit', required=False, type=int,
                              help="""If specified, only the templates containing more than the specified number of characters are used for the querying.""")

    query_parser.add_argument("-gt", "--ground-truth-folder", required=False, type=str,
                              help="""The path of the ground truth folder. If not specified, the ground truth is not used.""")

    args = parser.parse_args()

    if "template_text_limit" in vars(args) and args.template_text_limit and args.template_text_limit < 1:
        parser.error("The template text limit must be greater than 0.")

    return args
