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
import socket


# returns the Xeno Canto catalogue numbers for the given search terms.
# @param searchTerms: list of search terms
# http://www.xeno-canto.org/explore?query=common+snipe
# def read_numbers(search_terms):
#     i = 1  # page number
#     numbers = []
#     while True:
#         html = my_request('https://www.xeno-canto.org/explore?query={0}&pg={1}'.format(search_terms, i))
#         new_results = re.findall(r"/(\d+)/download", html)
#         if len(new_results) > 0:
#             numbers.extend(new_results)
#             print("read_numbers: the page: " + str(i))
#         # check if there are more than 1 page of results (30 results per page)
#         if len(new_results) < 30:
#             break
#         else:
#             i += 1  # move onto next page
#
#     return numbers

def html_downloader(s):
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

        proxy = {'https': '119.101.115.236:9999'}

        response = request.Request(s, headers=headers)
        # 使用ProxyHandler方法生成处理器对象
        proxy_handler = request.ProxyHandler(proxy)
        # 创建代理IP的opener实例
        opener = request.build_opener(proxy_handler)
        html = opener.open(response)
        # html = request.urlopen(response)
        html_utf8 = html.read().decode('utf-8')

    except error.HTTPError as e:
        if hasattr(e, 'code'):
            print('code: ' + str(e.code))
    except error.URLError as e:
        if hasattr(e, 'reason'):
            print('reason: ' + str(e.reason))
    else:
        print('Request for approval')
        return html_utf8


# returns the filenames for all Xeno Canto bird sound files found with the given
# search terms.
# @param searchTerms: list of search terms
def html_reader(search_terms):
    i = 1  # page number
    file_urls = []
    bird_names = []
    bird_ids = []
    while True:
        address = 'https://www.xeno-canto.org/explore?query={0}&pg={1}'.format(search_terms, i)
        print(address)
        html_utf8 = html_downloader(address)
        file_urls_results = re.findall(r"data-xc-filepath=\'(\S+)\'", html_utf8)
        bird_name_results = re.findall(r"class='scientific-name'>(.*?)</span>", html_utf8)
        id_results = re.findall(r"/(\d+)/download", html_utf8)

        if len(file_urls_results) > 0:
            file_urls.extend(file_urls_results)
            bird_names.extend(bird_name_results)
            bird_ids.extend(id_results)

            print("html_reader: page: " + str(i))
        # check if there are more than 1 page of results (30 results per page)
        if len(file_urls_results) < 30:
            break
        else:
            i += 1  # move onto next page

    return file_urls, bird_names, bird_ids


# creates the subdirectory sounds if necessary, and downloads all sound files
# found with the search terms into that directory. inserts the XC catalogue
# number in front of the file name, otherwise preserving original file names.
def crawling_scheduler(search_terms):
    # create sounds directory
    if not os.path.exists("sounds"):
        print("Creating subdirectory \"sounds\" for downloaded files...")
        os.makedirs("sounds")

    file_urls, bird_names, ids = html_reader(search_terms)

    if len(file_urls) == 0:
        print("No search results.")
        sys.exit()

    text_saver('log_files.txt', file_urls)
    text_saver('log_names.txt', bird_names)
    text_saver('log_ids.txt', ids)
    # numbers = read_numbers(searchTerms)

    # regex for extracting the filename from the file URL
    # fnFinder = re.compile('\S+/+(\S+)')
    print("A total of {0} files will be downloaded.".format(len(file_urls)))

    for i in range(0, len(file_urls)):
        # localFilename = numbers[i] + "_" + fnFinder.findall(file_names[i])[0]
        # some filenames in XC are in cyrillic characters, which causes them to
        # be too long in unicode values. in these cases, just use the ID number
        # as the filename.
        local_filename = bird_names[i] + "_" + ids[i] + ".mp3"
        if len("sounds/" + local_filename) > 255:
            local_filename = bird_names[i] + "_" + i + ".mp3"

        # 下载文件, 加入了重试机制
        print('downloading {}'.format(local_filename))
        if not os.path.exists("sounds/{}".format(bird_names[i])):
            print("Creating subdirectory /sounds/{}".format(bird_names[i]) + "for downloaded files...")
            os.makedirs("sounds/{}".format(bird_names[i]))
        try:
            request.urlretrieve('https:' + file_urls[i], "sounds/{}/".format(bird_names[i]) + local_filename,
                                _progress)
        except socket.timeout:
            count = 1
            while count <= 5:
                try:
                    request.urlretrieve('https:' + file_urls[i], "sounds/{}/".format(bird_names[i]) + local_filename,
                                        _progress)
                    break
                except socket.timeout:
                    err_info = 'Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count
                    print(err_info)
                    count += 1
            if count > 5:
                print("downloading {} failed!".format(file_urls[i]))

        except error.HTTPError as e:
            if hasattr(e, 'code'):
                print('code: ' + str(e.code) + '\n' + 'error url: ' + file_urls[i])
                continue
        except error.URLError as e:
            if hasattr(e, 'reason'):
                print('reason: ' + str(e.reason) + '\n' + 'error url: ' + file_urls[i])
                continue
        else:
            print("Downloading {} successful".format(local_filename))


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
    print("\rdownloading: %5.1f%%" % per, end="")
    # print(' {}%'.format(per))


def text_saver(filename, data):
    if not os.path.exists("logs"):
        print("Creating logs/...")
        os.makedirs("logs")

    if not os.path.exists("logs/{}".format(filename)):
        print("Creating log {}".format(filename))
        file = open('logs/'+filename, 'w')
        file.close()

    file = open('logs/'+filename, 'a')
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

def main():
    common = open("common.txt")
    for line in common:
        line = line.replace(' ', '+')
        line = line.strip('\n')
        print('for search {} :'.format(line))
        crawling_scheduler(line)

    common.close()

    # crawling_scheduler('box:25.791,110.799,30.194,115.677')


if __name__ == '__main__':
    main()
