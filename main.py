# A script to download bird sound files from the www.xeno-canto.org archives.
#
# forked from karoliina/xeno-canto-download
#
# The program downloads all the files found with the search terms into
# subdirectory sounds.

from urllib import request, error
import sys
import re
import os
import threading


# returns the Xeno Canto catalogue numbers for the given search terms.
# @param searchTerms: list of search terms
# http://www.xeno-canto.org/explore?query=common+snipe
def read_numbers(search_terms):
    i = 1  # page number
    numbers = []
    while True:
        html = my_request('https://www.xeno-canto.org/explore?query={0}&pg={1}'.format(search_terms, i))
        new_results = re.findall(r"/(\d+)/download", html)
        if len(new_results) > 0:
            numbers.extend(new_results)
            print("read_numbers: the page: " + str(i))
        # check if there are more than 1 page of results (30 results per page)
        if len(new_results) < 30:
            break
        else:
            i += 1  # move onto next page

    return numbers


# returns the filenames for all Xeno Canto bird sound files found with the given
# search terms.
# @param searchTerms: list of search terms
def read_filenames(search_terms):
    i = 1  # page number
    file_names = []
    while True:
        html = my_request('https://www.xeno-canto.org/explore?query={0}&pg={1}'.format(search_terms, i))
        new_results = re.findall(r"data-xc-filepath=\'(\S+)\'", html)
        if len(new_results) > 0:
            file_names.extend(new_results)
            print("read_filenames: the page: " + str(i))
        # check if there are more than 1 page of results (30 results per page)
        if len(new_results) < 30:
            break
        else:
            i += 1  # move onto next page

    return file_names


def my_request(s):
    try:
        headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                   # 'Accept_Encoding': 'gzip, deflate, br',
                   'Accept_Language': 'zh-CN,zh;q=0.9',
                   'Cache_Control': 'max-age=0',
                   'Connection': 'keep-alive',
                   # 'Cookie': '_utmz=47666734.1540381830.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none);
                   # __utma=47666734.732259021.1540381830.1542852869.1543043215.3;
                   # login=bWVldGRldmluLnpoQGdtYWlsLmNvbX4jfjlEaWp3RGd0WkNGNWl1VWU%3D;
                   # PHPSESSID=d6o3rccff8ac3u1p7n17cp1n45',
                   'DNT': '1',
                   'Host': 'www.xeno-canto.org',
                   'Upgrade_Insecure-Requests': '1',
                   'User_Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/69.0.3497.81 Safari/537.36'}

        response = request.Request(s, headers=headers)
        html = request.urlopen(response)
        result = html.read().decode('utf-8')

    except error.HTTPError as e:
        if hasattr(e, 'code'):
            print('code: ' + str(e.code))
    except error.URLError as e:
        if hasattr(e, 'reason'):
            print('reason: ' + str(e.reason))
    else:
        print('Request for approval')
        return result


# creates the subdirectory sounds if necessary, and downloads all sound files
# found with the search terms into that directory. inserts the XC catalogue
# number in front of the file name, otherwise preserving original file names.
def download(searchTerms):
    # create sounds directory
    if not os.path.exists("sounds"):
        print("Creating subdirectory \"sounds\" for downloaded files...")
        os.makedirs("sounds")
    file_names = read_filenames(searchTerms)
    if len(file_names) == 0:
        print("No search results.")
        sys.exit()
    text_save('list_of_download', file_names)

    numbers = read_numbers(searchTerms)
    text_save('list_of_numbers', numbers)

    # regex for extracting the filename from the file URL
    fnFinder = re.compile('\S+/+(\S+)')
    print("A total of {0} files will be downloaded.".format(len(file_names)))

    for i in range(0, len(file_names)):
        localFilename = numbers[i] + "_" + fnFinder.findall(file_names[i])[0]
        # some filenames in XC are in cyrillic characters, which causes them to
        # be too long in unicode values. in these cases, just use the ID number
        # as the filename.
        if len("sounds/" + localFilename) > 255:
            localFilename = numbers[i]

        try:
            print('downloading {}'.format(localFilename))
            request.urlretrieve('https:' + file_names[i], "sounds/" + localFilename, _progress)

        except error.HTTPError as e:
            if hasattr(e, 'code'):
                print('code: ' + str(e.code) + '\n' + 'error url: ' + file_names[i])
                continue
        except error.URLError as e:
            if hasattr(e, 'reason'):
                print('reason: ' + str(e.reason) + '\n' + 'error url: ' + file_names[i])
                continue
        else:
            print("Downloading {} successful".format(localFilename))


def _progress(block_num, block_size, total_size):
    """
    回调函数
    :param block_num: 已经下载的数据块
    :param block_size: 数据块的大小
    :param total_size: 远程文件的大小
    :return:
    """
    per = 100.0 * block_num * block_size / total_size
    if per > 100:
        per = 100
    print(' {}%'.format(per))


def text_save(filename, data):
    file = open(filename, 'a')
    for i in range(len(data)):
        # 去除[],这两行按数据不同，可以选择
        s = str(data[i]).replace('[', '').replace(']', '')
        # 去除单引号，逗号，每行末尾追加换行符
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("{} 保存成功".format(data))


# def main(argv):
#     if len(sys.argv) < 2:
#         print("Usage: python xcdl.py searchTerm1 searchTerm2 ... searchTermN")
#         print("Example: python xcdl.py common snipe")
#         return
#     else:
#         download(argv[1:len(argv)])
#
#
# if __name__ == "__main__":
#
#     snipe = "common.txt"
#
#     main(snipe)


download('box:20.036,102.79,34.471,125.29')
