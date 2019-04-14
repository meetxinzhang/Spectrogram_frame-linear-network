# 一个从 www.xeno-canto.org 下载音频数据的网络爬虫
#
# 下载列表维护，支持断点续传；
# 捕获网络异常，加入重试机制；
# 可根据重试计数更换代理；
# 模块化设计，可根据文件计数器加入动态代理；
#
# 不同的声音种类放入 mp3/ 下不同的文件夹，文件夹名直接就是标签
# 文件列表放入 logs/ 文件夹下
#
# by Devin Zhang
# meetdevin.zh@outlook.com
#
#
# 可用代理网站
# http://www.xicidaili.com/
# http://www.66ip.cn/
# http://www.mimiip.com/gngao/
# http://www.kuaidaili.com/

from urllib import request, error
import sys
import re
import os
import socket


class MyWebCrawler:
    socket.setdefaulttimeout(100)
    proxy = {'https': '39.107.84.185:8123'}
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
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

    def __init__(self):
        if not os.path.exists("mp3"):
            print("Creating subdirectory \"mp3\" for downloaded files...")
            os.makedirs("mp3")

        # if not os.path.exists("logs/" + search_terms):
        #     print("Creating logs/{}...".format(search_terms))
        #     os.makedirs("logs/" + search_terms)
        pass

    def html_downloader(self, s):
        try:
            response = request.Request(s, headers=self.headers)
            # # 使用ProxyHandler方法生成处理器对象
            # proxy_handler = request.ProxyHandler(self.proxy)
            # # 创建代理IP的opener实例
            # opener = request.build_opener(proxy_handler)
            # html = opener.open(response)
            html = request.urlopen(response)
            html_utf8 = html.read().decode('utf-8')

        except error.HTTPError as e:
            if hasattr(e, 'code'):
                print('html_downloader_code: ' + str(e.code))
        except error.URLError as e:
            if hasattr(e, 'reason'):
                print('html_downloader_reason: ' + str(e.reason))
        except socket.timeout:
            count = 1
            while count <= 5:
                try:
                    response = request.Request(s, headers=self.headers)
                    # 使用ProxyHandler方法生成处理器对象
                    proxy_handler = request.ProxyHandler(self.proxy)
                    # 创建代理IP的opener实例
                    opener = request.build_opener(proxy_handler)
                    html = opener.open(response)
                    # html = request.urlopen(response)
                    html_utf8 = html.read().decode('utf-8')
                    return html_utf8
                except socket.timeout:
                    err_info = 'Request for %d time' % count if count == 1 else 'Request for %d times' % count
                    print(err_info)
                    count += 1
            if count > 5:
                print("Request  failed!")
        else:
            print('Request for approval')
            return html_utf8

    # returns the filenames for all Xeno Canto bird sound files found with the given
    # search terms.
    # @param searchTerms: list of search terms
    def html_reader(self, search_terms):
        i = 1  # page number
        file_urls = []
        bird_names = []
        bird_ids = []
        while True:
            address = 'https://www.xeno-canto.org/explore?query={0}&pg={1}'.format(search_terms, i)
            print(address)
            html_utf8 = self.html_downloader(address)
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

    # creates the subdirectory mp3 if necessary, and downloads all sound files
    # found with the search terms into that directory. inserts the XC catalogue
    # number in front of the file name, otherwise preserving original file names.
    def crawling_scheduler(self, search_terms):
        if os.path.exists('logs/' + search_terms):
            file_urls = self.logs_recover('log_files.txt', search_terms)
            bird_names = self.logs_recover('log_names.txt', search_terms)
            ids = self.logs_recover('log_ids.txt', search_terms)
            print('从 logs/{} 中恢复日志数据'.format(search_terms))
        else:
            file_urls, bird_names, ids = self.html_reader(search_terms)

            if len(file_urls) == 0:
                print("No search results.")
                sys.exit()
            else:
                self.logs_saver(folder=search_terms, filename='log_files.txt', data=file_urls)
                self.logs_saver(folder=search_terms, filename='log_names.txt', data=bird_names)
                self.logs_saver(folder=search_terms, filename='log_ids.txt', data=ids)

        # regex for extracting the filename from the file URL
        # fnFinder = re.compile('\S+/+(\S+)')
        print("A total of {0} files will be downloaded.".format(len(file_urls)))

        for i in range(0, len(file_urls)):
            # localFilename = numbers[i] + "_" + fnFinder.findall(file_names[i])[0]
            # some filenames in XC are in cyrillic characters, which causes them to
            # be too long in unicode values. in these cases, just use the ID number
            # as the filename.
            local_filename = bird_names[i] + "_" + ids[i] + ".mp3"
            if len("mp3/" + local_filename) > 255:
                local_filename = bird_names[i] + "_" + str(i) + ".mp3"

            if not os.path.exists("mp3/{}/{}".format(search_terms, bird_names[i])):
                # 如果没有该分类文件夹，则创建文件夹用于保存该分类下文件
                print("########## Creating subdirectory /mp3/{}/{} "
                      .format(search_terms, bird_names[i]) + "for downloaded files...")
                os.makedirs("mp3/{}/{}".format(search_terms, bird_names[i]))
            if os.path.exists("mp3/{}/{}/{}".format(search_terms, bird_names[i], local_filename)):
                print('{} checked {}'.format(i, local_filename))
                # 如果该文件已经存在，则继续下一次循环
                continue

            # 下载文件, 利用 socket.timeout 加入了重试机制
            print('{} downloading {}'.format(i, local_filename))
            self.downloader_retry('https:' + file_urls[i],
                                  "mp3/{}/{}/{}".format(search_terms, bird_names[i], local_filename))

            # try:
            #     request.urlretrieve('https:' + file_urls[i], "mp3/{}/".format(bird_names[i]) + local_filename,
            #                         self._progress)
            # except socket.timeout:
            #     count = 1
            #     while count <= 5:
            #         try:
            #             request.urlretrieve('https:' + file_urls[i],
            #                                 "mp3/{}/".format(bird_names[i]) + local_filename,
            #                                 self._progress)
            #             break
            #         except socket.timeout:
            #             err_info = 'Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count
            #             print(err_info)
            #             count += 1
            #     if count > 5:
            #         print("downloading {} failed!".format(file_urls[i]))
            #         self.finished = False
            #
            # except error.HTTPError as e:
            #     if hasattr(e, 'code'):
            #         print('code: ' + str(e.code) + '\n' + 'error url: ' + file_urls[i])
            #         self.finished = False
            #         continue
            # except error.URLError as e:
            #     if hasattr(e, 'reason'):
            #         print('reason: ' + str(e.reason) + '\n' + 'error url: ' + file_urls[i])
            #         self.finished = False
            #         continue
            # else:
            #     print("Downloading {} successful".format(local_filename))

    def downloader_retry(self, url, local_file):
        count = 1
        while count <= 5:
            try:
                request.urlretrieve(url, local_file, self._progress)
                print(" Download {} successful".format(local_file))
                break
            except socket.timeout:
                print('!!!!!!!!!!downloader_retry_timeout')
                print('Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count)
                count += 1
            except error.HTTPError as e:
                if hasattr(e, 'code'):
                    print('!!!!!!!!!!downloader_retry_code: ' + str(e.code) + '\n' + 'error url: ' + url)
                    print('Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count)
                    count += 1
            except error.URLError as e:
                if hasattr(e, 'reason'):
                    print('!!!!!!!!!!downloader_retry_reason: ' + str(e.reason) + '\n' + 'error url: ' + url)
                    print('Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count)
                    count += 1
        if count > 5:
            print("!!!!!!!!!!download failed!!!!!!!!!! for {}".format(url))

    def _progress(self, block_num, block_size, total_size):
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
        print("\r progress: %5.1f%%" % per, end="")
        # print(' {}%'.format(per))

    def logs_saver(self, folder, filename, data):
        if not os.path.exists("logs/" + folder):
            print("########## Creating logs/{}...".format(folder))
            os.makedirs("logs/" + folder)

        if not os.path.exists("logs/{}/{}".format(folder, filename)):
            print("########## Creating log/{}/{}".format(folder, filename))
            file = open('logs/{}/{}'.format(folder, filename), 'w')
            file.close()

        file = open('logs/{}/{}'.format(folder, filename), 'a')
        for i in range(len(data)):
            # 去除[],这两行按数据不同，可以选择
            s = str(data[i]).replace('[', '').replace(']', '')
            # 去除单引号，逗号，每行末尾追加换行符
            s = s.replace("'", '').replace(',', '') + '\n'
            file.write(s)
        file.close()
        print("保存成功:{}".format(data))

    def logs_recover(self, filename, search_terms):
        list = []
        file = open('logs/{}/{}'.format(search_terms, filename))
        for line in file:
            line = line.strip('\n')
            list.append(line)
        file.close()
        return list


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

# crawling_scheduler('box:25.791,110.799,30.194,115.677')


if __name__ == '__main__':
    commons = [
        # 'Cuculus micropterus',  # 107
        # 'Pica pica'  # 450
        # 'Oriolus oriolus',  # 539
        # 'Anser anser',  # 337
        # 'Phasianus colchicus'  # 299
        # 'Tachybaptus ruficollis'  # 286
        # 'Ardea cinerea',  # 339
        # 'Accipiter gentilis',  # 262
        # 'Buteo buteo',  # 395
    ]

    # commons = open("common.txt")
    mwc = MyWebCrawler()

    for line in commons:
        # if line is '\n':
        #     continue

        line = line.replace(' ', '+')
        # line = line.strip('\n')
        # line = ''.join(list(filter(lambda i: not i.isdigit(), line)))
        print('\nfor search {} :'.format(line))
        mwc.crawling_scheduler(line)

    # commons.close()
