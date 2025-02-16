import os


def list_directories(base_path):
    """读取当前目录的所有目录，返回所有目录的路径"""
    # 获取 base_path 下的所有一级目录
    first_level_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # 存储二级目录下的一级目录
    second_level_dirs = []
    for first_level_dir in first_level_dirs:
        first_level_path = os.path.abspath(os.path.join(base_path, first_level_dir))
        second_level_dirs.append(first_level_path)
    return second_level_dirs


def list_files(base_path):
    """读取当前目录的所有.jpg文件"""
    en_images_list = []
    all_files = os.listdir(base_path)
    jpg_files = [file for file in all_files if file.endswith('.jpg')]
    for file in jpg_files:
        en_images_list.append(file.strip())
    return en_images_list


def get_sparse_neighbor(p: int, n: int, m: int):
    """Returns a dictionnary, where the keys are index of 4-neighbor of `p` in the sparse matrix,
       and values are tuples (i, j, x), where `i`, `j` are index of neighbor in the normal matrix,
       and x is the direction of neighbor.

    Arguments:
        p {int} -- index in the sparse matrix.
        n {int} -- number of rows in the original matrix (non sparse).
        m {int} -- number of columns in the original matrix.

    Returns:
        dict -- dictionnary containing indices of 4-neighbors of `p`.
    """
    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d


if __name__ == '__main__':
    list_dirs = list_directories(r'F:\LLimg\ELLI\output\pic')
    for dir in list_dirs:
        print(dir)

    print(list_files(list_dirs[0]))
