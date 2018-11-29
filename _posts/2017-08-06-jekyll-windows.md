---
layout: post
title:  Jekyll的Windows环境配置
subtitle:  个人博客搭建回顾
date: 2017-08-6 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-jekyll.jpg"
catalog: true
tags:
    - 博客
---


Jekyll是一个简单的免费的Blog生成工具，类似WordPress。但是和WordPress又有很大的不同，原因是jekyll只是一个生成静态网页的工具，不需要数据库支持。但是可以配合第三方服务,例如Disqus。最关键的是jekyll可以免费部署在Github上，而且可以绑定自己的域名。Jekyll的配置其实非常简单，但因为每次重装系统都要再弄一次，因此这次我还是记录下来，省的下次重装系统后忘记什么还要再查。

Jekyll是基于ruby开发的，因此，需要先安装ruby，ruby安装很简单，直接下载[Ruby installer](https://rubyinstaller.org/downloads/)即可。在安装好rubyinstaller后可以通过命令ruby -v检测是否安装成功。
![](/img/in-post/post-jekyll/jekyll-1.png)
之后，关掉命令窗口，然后再打开，重新输入命令gem install jekyll
![](/img/in-post/post-jekyll/jekyll-3.png)

到此我以为自己装完了jekyll，但是，运行jekyll serve后，却出现如下错误
![](/img/in-post/post-jekyll/jekyll-4.png)

上网查询后知道，需在命令行窗口输入gem install jekyll-paginate，再安装一个组件才可成功。

至此，jekyll的环境配置已完成，使用时只需到相应目录，在命令行窗口中输入jekyll serve即可。
