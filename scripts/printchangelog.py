import sys


def get_latest_tag(changelog):
    with open(changelog, "r") as file:
        lines = file.readlines()

    tags = []
    for index, line in enumerate(lines):
        if line.startswith("# "):
            tags.append((index, line.strip("# ").strip()))

    if not tags:
        print("No tags found.")
        return

    start_index, latest_tag = tags[0]
    end_index = tags[1][0] if len(tags) > 1 else len(lines)
    content = "".join(lines[start_index:end_index])

    print(content)


if __name__ == "__main__":
    changelog_file = "CHANGELOG.md"
    if len(sys.argv) > 1:
        changelog_file = sys.argv[1]
    get_latest_tag(changelog_file)
