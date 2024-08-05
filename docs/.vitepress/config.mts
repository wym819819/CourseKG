import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "CourseKG",
  description: "使用大模型自动构建课程知识图谱",
  head: [['link', { rel: 'icon', href: '/logo.png' }]],
  themeConfig: {
    search: {
      provider: 'local',
    },
    nav: [
      { text: '主页', link: '/' },
      { text: '示例', link: '/examples' }
    ],
    logo: '/logo.png',

    sidebar: [
      {
        text: '目录',
        items: [
          { text: '示例', link: '/examples' },
          { text: 'API 参考', link: '/api-reference' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wangtao2001/CourseKG' }
    ]
  }
})
