# [Qi Personal Website](https://qw2261.github.io/)

## 开发环境

### 前置依赖

| 依赖 | 安装方式 |
|-----|---------|
| nvm | `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh \| bash`（或使用国内镜像：`curl -o- https://cdn.jsdelivr.net/gh/nvm-sh/nvm@v0.39.7/install.sh \| bash`） |
| Node.js 20 | `nvm install 20 && nvm alias default 20`（如需加速：`export NVM_NODEJS_ORG_MIRROR=https://mirrors.tencent.com/nodejs-release/`） |

### 包管理

项目使用 **npm**，所有操作在 `blog/` 根目录中执行。

```bash
cd blog
npm install        # 安装依赖（仅首次或 package.json 变更后）
```

### 启动开发服务器

```bash
cd blog
npm run dev        # 启动后访问 http://localhost:5173
```

### 构建生产版本

```bash
cd blog
npm run build      # 输出到 dist/
npm run preview    # 本地预览构建产物
```

### 写博客

博客文章使用 Markdown 格式，存储在 `src/data/posts/` 目录下。

**新增博客步骤：**

1. 在 `src/data/posts/` 目录新建 `.md` 文件，例如 `new-post.md`：
   ```markdown
   ## 我的新博客标题

   2026-05-04，于北京

   这是我的第一篇用 Markdown 写的博客！

   支持 **加粗**、*斜体*、~~删除线~~。

   ### 子标题

   - 无序列表项1
   - 无序列表项2

   ![图片说明](/img/xxx.jpeg)

   [链接文字](https://example.com)
   ```

2. 在 `src/data/posts.js` 中添加元数据：
   ```javascript
   {
     id: 'new-post',       // 必须和文件名一致（不含 .md）
     title: '我的新博客标题',
     date: '2026-05-04'
   }
   ```

### 目录结构（最终状态 M7）

```
blog/
├── public/                # 不编译的静态资源（PDF、favicon、图片）
│   ├── file/              # PDF 文件
│   └── img/               # 静态图片
├── src/
│   ├── assets/            # 编译的资源（样式、图片）
│   ├── components/        # 组件（Header、Footer、Sidebar、Dropdown、BlogCard）
│   ├── data/              # 数据文件
│   │   ├── posts/         # Markdown 博客文章
│   │   │   ├── noscience.md
│   │   │   ├── shanghai.md
│   │   │   └── ...
│   │   ├── posts.js       # 博客元数据
│   │   └── navigation.js  # 导航数据
│   ├── router/            # 路由配置
│   └── views/             # 页面组件（Home、Blog、BlogPost、Project、Bio、Share、TechBook、GroupProject）
├── index.html             # Vite 入口
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── package.json
├── .gitignore
├── dist/                  # 构建产物（gitignore）
└── README.md
```

### 工作进度

| 里程碑 | 状态 | 完成日期 | 说明 |
|-------|------|---------|------|
| M0 脚手架搭建 | ✅ 已完成 | 2026-05-03 | Vue3 + Vite + Tailwind CSS + Vue Router，开发服务器和构建均正常 |
| M1 布局 + 导航 | ✅ 已完成 | 2026-05-03 | Header / Footer / Sidebar / Dropdown 组件，SPA 路由跳转，响应式移动端适配 |
| M2 逐页迁移内容 | ✅ 已完成 | 2026-05-03 | 6 页内容全部迁移，图片/PDF/iframe 路径一致，构建通过 65 modules |
| M3 博客系统 | ✅ 已完成 | 2026-05-03 | 5 篇博文数据驱动渲染，Blog 列表 + BlogPost 详情，PT Serif 排版 |
| M4 构建验证 | ✅ 已完成 | 2026-05-03 | dist/ 独立可运行，23 个静态资源完整，GH Pages 兼容 |
| M5 部署上线 | ✅ 已完成 | 2026-05-04 | GitHub Actions 自动部署到 gh-pages，旧链接 404 重定向，线上全页可访问 |
| M6 清理收尾 | ✅ 已完成 | 2026-05-04 | Vue 应用提升到根目录，删除旧 Jekyll 文件，更新 GitHub Actions |
| M7 博客系统优化 | ✅ 已完成 | 2026-05-04 | 支持 Markdown 写博客，文章内容存为独立 .md 文件 |

---

## 重构计划：Vue3 + Vite + Tailwind CSS + GitHub Pages

> ⬇️ 以下为完整的迁移方案设计文档，供开发参考。

### 背景

当前站点基于 Jekyll 主题，使用纯 HTML + CSS + jQuery 1.9.1 构建。页面间存在大量重复代码（导航栏、页脚、侧边栏），依赖过时的 IE 兼容脚本和旧版 jQuery，新增博客文章需手动编辑 HTML，可维护性差。

目标：迁移到 Vue3 + Vite，生成纯静态文件部署到 GitHub Pages，实现组件化、数据驱动、自动化部署。

---

### 技术栈

| 工具 | 版本 | 作用 |
|-----|------|------|
| Node.js | ^20.x | 运行时环境（本地开发与 GitHub Actions 均需） |
| Vue 3 | ^3.x | 前端框架，Composition API + `<script setup>` |
| Vite | ^6.x | 构建工具，极速冷启动 + 按需编译 |
| Tailwind CSS | ^3.x | 原子化 CSS 框架，无需手写样式文件 |
| Vue Router | ^4.x | 前端路由，使用 `createWebHashHistory`（适配 GitHub Pages） |
| GitHub Actions | - | 自动构建 & 部署到 gh-pages 分支（见 `deploy.yml`） |

### 项目初始化

> ⚠️ **注意**：采用 `vue-app/` 子目录方案（见下方"渐进式迁移策略"），不直接在 `blog/` 根目录初始化。以下命令已包含在 M0 里程碑中。

```bash
# 创建子目录并初始化（已在 M0 中执行）
mkdir vue-app
cd vue-app
npm create vite@latest . -- --template vue
npm install
npm install tailwindcss@3 postcss autoprefixer vue-router@4
npx tailwindcss init -p
```

---

### 目录结构

#### 开发阶段（M0-M5）

Vue 应用在 `vue-app/` 子目录中开发，旧站文件保持不动。构建产物输出到 `blog/dist/`。

```
blog/
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions（working-directory: vue-app）
├── file/                           # 旧站 PDF 文件（迁移期间不动）
│   ├── Qi Wang.pdf
│   ├── 王琦 简历.pdf
│   └── bigdata.pdf
├── img/                            # 旧站图片（迁移期间不动）
│   ├── head.jpeg
│   ├── sea.png
│   ├── title-icon.png
│   └── ...
├── html/                           # 旧 HTML 页面（M6 删除）
├── css/                            # 旧样式（M6 删除）
├── js/                             # 旧脚本（M6 删除）
├── fonts/                          # 旧字体（M6 删除）
├── photo/                          # 旧照片（M6 删除）
├── _config.yml                     # Jekyll 配置（M6 删除）
├── index.html                      # 旧站首页（M6 删除）
├── vue-app/                        # ★ 新 Vue 应用
│   ├── public/                    # Vite 不编译的静态资源
│   │   ├── file/                  # PDF（从 blog/file/ 复制）
│   │   │   ├── Qi Wang.pdf
│   │   │   ├── 王琦 简历.pdf
│   │   │   └── bigdata.pdf
│   │   ├── img/                   # 图标（从 blog/img/ 复制）
│   │   │   └── title-icon.png
│   │   └── 404.html               # 旧链接重定向（仅 History 模式需要，Hash 模式可删除）
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.vue
│   │   │   ├── Footer.vue
│   │   │   ├── Sidebar.vue
│   │   │   ├── BlogCard.vue
│   │   │   └── Dropdown.vue
│   │   ├── views/
│   │   │   ├── Home.vue
│   │   │   ├── Blog.vue
│   │   │   ├── BlogPost.vue
│   │   │   ├── Project.vue
│   │   │   ├── Bio.vue
│   │   │   ├── Share.vue
│   │   │   ├── TechBook.vue
│   │   │   └── GroupProject.vue
│   │   ├── data/
│   │   │   ├── posts.js
│   │   │   └── navigation.js
│   │   ├── router/
│   │   │   └── index.js
│   │   ├── assets/
│   │   │   ├── images/            # 从 blog/img/ 复制
│   │   │   │   ├── head.jpeg
│   │   │   │   ├── sea.png
│   │   │   │   ├── github-icon.png
│   │   │   │   ├── linkedin-icon.jpg
│   │   │   │   └── email-icon.png
│   │   │   └── styles/
│   │   │       └── main.css
│   │   ├── App.vue
│   │   └── main.js
│   ├── index.html
│   ├── vite.config.js            # outDir: '../dist'
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── package.json
│   └── .gitignore                # 忽略 node_modules/
├── dist/                          # ★ 构建产物（gitignore）
└── README.md
```

#### M6 完成后（最终结构）

M6 将 `vue-app/` 内容提升到根目录，删除所有旧文件。`vite.config.js` 的 `outDir` 改回 `'dist'`。

```
blog/
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions（标准根目录模式）
├── public/                         # Vite 不编译的静态资源
│   ├── file/
│   │   ├── Qi Wang.pdf
│   │   ├── 王琦 简历.pdf
│   │   └── bigdata.pdf
│   ├── img/
│   │   └── title-icon.png
│   └── 404.html
├── src/
│   ├── components/
│   │   ├── Header.vue
│   │   ├── Footer.vue
│   │   ├── Sidebar.vue
│   │   ├── BlogCard.vue
│   │   └── Dropdown.vue
│   ├── views/
│   │   ├── Home.vue
│   │   ├── Blog.vue
│   │   ├── BlogPost.vue
│   │   ├── Project.vue
│   │   ├── Bio.vue
│   │   ├── Share.vue
│   │   ├── TechBook.vue
│   │   └── GroupProject.vue
│   ├── data/
│   │   ├── posts.js
│   │   └── navigation.js
│   ├── router/
│   │   └── index.js
│   ├── assets/
│   │   ├── images/            # 从 blog/img/ 复制
│   │   │   ├── head.jpeg
│   │   │   ├── sea.png
│   │   │   ├── github-icon.png
│   │   │   ├── linkedin-icon.jpg
│   │   │   └── email-icon.png
│   │   └── styles/
│   │       └── main.css
│   ├── App.vue
│   └── main.js
├── index.html
├── vite.config.js                  # outDir: 'dist'
├── tailwind.config.js
├── postcss.config.js
├── package.json
├── .gitignore
├── dist/                           # 构建产物（gitignore）
└── README.md
```

> **关键变化**：M6 提升到根目录后，`vite.config.js` 的 `outDir` 从 `'../dist'` 改为 `'dist'`，GitHub Actions 去掉 `working-directory: vue-app`，触发分支从 `vue-migration` 改为 `main`。

---

### 关键文件配置

#### `vite.config.js`

开发阶段（M0-M5），构建产物输出到上级目录：

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  base: '/',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true
  }
})
```

> M6 提升到根目录后，`outDir` 改为 `'dist'`，`emptyOutDir: true` 保持不变。

#### `deploy.yml` - GitHub Actions 部署配置

完整内容见下方"M5：部署切换"及"部署方案 → 阶段二"章节，分别对应 `vue-migration` 分支（开发阶段）和 `main` 分支（M6 后）的配置。

#### `tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 可自定义品牌色
        primary: '#2c3e50',
        accent: '#42b883'
      },
      fontFamily: {
        sans: ['PT Sans Narrow', 'sans-serif'],
        serif: ['PT Serif', 'serif']
      }
    },
  },
  plugins: [],
}
```

#### `postcss.config.js`

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

#### `src/assets/styles/main.css` - Tailwind 入口

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

#### `src/main.js` - 入口文件

```javascript
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './assets/styles/main.css'

const app = createApp(App)
app.use(router)
app.mount('#app')
```

#### `index.html` - Vite 入口

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Qi Wang's Website" />
  <meta name="keywords" content="CityU, Columbia, Robotics, Qi Wang" />
  <link rel="icon" type="image/png" href="/img/title-icon.png" sizes="32x32" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=PT+Sans+Narrow:wght@400;700&family=PT+Serif:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet" />
  <title>Qi's Site</title>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

---

### 路由设计

#### 方案选择：Hash 模式（推荐）

由于部署在 GitHub Pages，**推荐使用 `createWebHashHistory`** 模式，URL 形如 `/#/blog/noscience`。这样 GitHub Pages 会将所有请求都解析到 `index.html`，由前端路由自行处理，无需自定义 404 页面或 JavaScript 重定向。

`src/router/index.js`：

```javascript
import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue')
  },
  {
    path: '/blog',
    name: 'Blog',
    component: () => import('@/views/Blog.vue')
  },
  {
    path: '/blog/:id',
    name: 'BlogPost',
    component: () => import('@/views/BlogPost.vue'),
    props: true
  },
  {
    path: '/project',
    name: 'Project',
    component: () => import('@/views/Project.vue')
  },
  {
    path: '/bio',
    name: 'Bio',
    component: () => import('@/views/Bio.vue')
  },
  {
    path: '/share',
    name: 'Share',
    component: () => import('@/views/Share.vue')
  },
  {
    path: '/studys/techbook',
    name: 'TechBook',
    component: () => import('@/views/TechBook.vue')
  },
  {
    path: '/project/iot',
    name: 'GroupProject',
    component: () => import('@/views/GroupProject.vue')
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/'
  }
]

const router = createRouter({
  // ★ 使用 Hash 模式，GitHub Pages 原生支持，无需 404 重定向
  history: createWebHashHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  }
})

export default router
```

#### 备选方案：History 模式（需配置 404.html）

如果希望 URL 不带 `#`（如 `/blog/noscience`），可以使用 `createWebHistory`，但需要额外配置 `public/404.html` 做路由回退，且需确保 GitHub Pages 的 Custom 404 指向该文件。此方案在某些浏览器/GitHub Pages 版本下可能出现短暂 404 闪烁，**不建议用于生产环境**。

**路由与旧页面对应关系：**

| 旧路径 | 新路由 |
|-------|--------|
| `/index.html` | `/` |
| `/html/blog.html` | `/blog` |
| `/html/project.html` | `/project` |
| `/html/bio.html` | `/bio` |
| `/html/share.html` | `/share` |
| `/html/studys/TechBook.html` | `/studys/techbook` |
| `/html/Group_3_Project_Website/index.html` | `/project/iot` |
| `/html/blogs/noscience.html` 等 | `/blog/:id`（动态路由） |

---

### 组件设计

#### `App.vue` - 根组件（Layout）

```vue
<template>
  <Header />
  <div class="image-wrap">
    <img src="@/assets/images/sea.png" alt="feature image" class="w-full" />
  </div>
  <div class="max-w-4xl mx-auto px-4 py-8 flex flex-col md:flex-row gap-8">
    <Sidebar />
    <main class="flex-1">
      <router-view />
    </main>
  </div>
  <Footer />
</template>

<script setup>
import Header from '@/components/Header.vue'
import Footer from '@/components/Footer.vue'
import Sidebar from '@/components/Sidebar.vue'
</script>
```

#### `src/components/Header.vue` - 导航栏

```vue
<template>
  <header class="bg-white border-b border-gray-200 sticky top-0 z-50">
    <div class="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
      <a href="/" class="text-xl font-bold text-gray-800 hover:text-gray-600">
        Qi Wang's site
      </a>
      <nav class="hidden md:flex items-center space-x-6">
        <router-link to="/" class="nav-link">Home</router-link>
        <router-link to="/project" class="nav-link">Projects</router-link>
        <router-link to="/blog" class="nav-link">Blog</router-link>
        <Dropdown title="MY STORIES" :items="storiesItems" />
        <Dropdown title="CV" :items="cvItems" />
      </nav>
      <button class="md:hidden" @click="mobileOpen = !mobileOpen">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
    </div>
    <!-- 移动端菜单（包含下拉菜单） -->
    <div v-if="mobileOpen" class="md:hidden bg-white border-t px-4 py-3 space-y-1">
      <router-link to="/" class="block nav-link py-2" @click="mobileOpen = false">Home</router-link>
      <router-link to="/project" class="block nav-link py-2" @click="mobileOpen = false">Projects</router-link>
      <router-link to="/blog" class="block nav-link py-2" @click="mobileOpen = false">Blog</router-link>
      <!-- MY STORIES 下拉（移动端） -->
      <div class="py-2">
        <div class="nav-link font-semibold text-gray-700">MY STORIES</div>
        <div class="pl-4 mt-1 space-y-1">
          <router-link to="/bio" class="block text-sm text-gray-600 py-1" @click="mobileOpen = false">Biography</router-link>
          <router-link to="/share" class="block text-sm text-gray-600 py-1" @click="mobileOpen = false">Share</router-link>
          <router-link to="/studys/techbook" class="block text-sm text-gray-600 py-1" @click="mobileOpen = false">TechBook</router-link>
          <router-link to="/project/iot" class="block text-sm text-gray-600 py-1" @click="mobileOpen = false">IoT Smart Movable Trash Bin Project</router-link>
          <a href="/file/bigdata.pdf" target="_blank" class="block text-sm text-gray-600 py-1">Big Data Analytics Paper</a>
        </div>
      </div>
      <!-- CV 下拉（移动端） -->
      <div class="py-2">
        <div class="nav-link font-semibold text-gray-700">CV</div>
        <div class="pl-4 mt-1 space-y-1">
          <a href="/file/Qi Wang.pdf" target="_blank" class="block text-sm text-gray-600 py-1">Qi Wang's CV</a>
          <a href="/file/王琦 简历.pdf" target="_blank" class="block text-sm text-gray-600 py-1">王琦 简历</a>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref } from 'vue'
import Dropdown from './Dropdown.vue'
import { storiesItems, cvItems } from '@/data/navigation'

const mobileOpen = ref(false)
</script>
```

#### `src/components/Footer.vue` - 页脚

```vue
<template>
  <footer class="border-t border-gray-200 mt-12 py-6 text-center text-sm text-gray-500">
    <span>&copy; {{ year }} Qi Wang. Powered by Vue &amp; Vite.</span>
    <div class="flex justify-center space-x-4 mt-2">
      <a href="https://github.com/qw2261" target="_blank" class="hover:text-gray-700">GitHub</a>
      <a href="https://www.linkedin.com/in/qwangmatt/" target="_blank" class="hover:text-gray-700">LinkedIn</a>
      <a href="mailto:qw2261@columbia.edu" class="hover:text-gray-700">Email</a>
    </div>
  </footer>
</template>

<script setup>
const year = new Date().getFullYear()
</script>
```

#### `src/components/Sidebar.vue` - 侧边栏

```vue
<template>
  <aside class="w-full md:w-64 shrink-0 text-center">
    <img src="@/assets/images/head.jpeg" alt="Qi Wang" class="w-32 h-32 rounded-full mx-auto" />
    <h3 class="mt-4 text-lg font-semibold">Qi Wang</h3>
    <p class="text-gray-600 mt-2">步履虽慢，未曾折返</p>
    <p class="text-gray-600">私の歩みは遅いが、歩んだ道を引き返すことはない</p>
    <div class="mt-4 space-y-1">
      <a href="https://github.com/qw2261" target="_blank" class="block text-sm text-gray-500 hover:text-gray-700">
        <img src="@/assets/images/github-icon.png" class="inline w-5 h-5 mr-2" />Github
      </a>
      <a href="https://www.linkedin.com/in/qwangmatt/" target="_blank" class="block text-sm text-gray-500 hover:text-gray-700">
        <img src="@/assets/images/linkedin-icon.jpg" class="inline w-5 h-5 mr-2" />LinkedIn
      </a>
      <a href="mailto:qw2261@columbia.edu" class="block text-sm text-gray-500 hover:text-gray-700">
        <img src="@/assets/images/email-icon.png" class="inline w-5 h-5 mr-2" />Email
      </a>
    </div>
  </aside>
</template>
```

#### `src/components/Dropdown.vue` - 下拉菜单

```vue
<template>
  <div
    class="relative"
    @mouseenter="open = true"
    @mouseleave="open = false"
    @click="open = !open"
  >
    <button class="nav-link font-semibold flex items-center gap-1 cursor-pointer">
      {{ title }}
      <span class="text-xs transition-transform" :class="open ? 'rotate-180' : ''">▼</span>
    </button>
    <div
      v-show="open"
      class="absolute top-full left-0 mt-1 bg-white border rounded shadow-lg py-1 min-w-[200px] z-50"
    >
      <template v-for="item in items" :key="item.label">
        <router-link
          v-if="!item.external"
          :to="item.to"
          class="block px-4 py-2 text-sm hover:bg-gray-100"
          @click="open = false"
        >
          {{ item.label }}
        </router-link>
        <a
          v-else
          :href="item.href"
          target="_blank"
          rel="noopener noreferrer"
          class="block px-4 py-2 text-sm hover:bg-gray-100"
        >
          {{ item.label }}
        </a>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  title: String,
  items: Array
})

const open = ref(false)
</script>
```

#### `src/components/BlogCard.vue` - 博客列表卡片

保留旧站样式，对应旧 `<div class="blog-title">` 结构：

```vue
<template>
  <div class="blog-title border-b border-gray-100 py-3">
    <div class="blog-time text-gray-400 text-sm">{{ dateDisplay }}</div>
    <div class="blog-name">
      <router-link
        :to="`/blog/${post.id}`"
        class="text-gray-700 hover:text-gray-900 underline underline-offset-2"
      >
        {{ post.title }}
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  post: {
    type: Object,
    required: true
  }
})

const dateDisplay = computed(() => {
  const d = new Date(props.post.date)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
})
</script>
```

---

### 数据驱动设计

#### `src/data/navigation.js` - 导航数据

```javascript
export const storiesItems = [
  { label: 'Biography', to: '/bio' },
  { label: 'Share', to: '/share' },
  { label: 'TechBook', to: '/studys/techbook' },
  { label: 'IoT Smart Movable Trash Bin Project', to: '/project/iot' },
  { label: 'Big Data Analytics Paper', href: '/file/bigdata.pdf', external: true }
]

export const cvItems = [
  { label: "Qi Wang's CV", href: '/file/Qi Wang.pdf', external: true },
  { label: '王琦 简历', href: '/file/王琦 简历.pdf', external: true }
]
```

#### `src/data/posts.js` - 博客文章数据

```javascript
export const posts = [
  {
    id: 'noscience',
    title: '没有科学的世界 (一）',
    date: '2018-07-11',
    content: `<!-- 文章 HTML 内容 -->`,
    summary: '第一章简介...'
  },
  {
    id: 'shanghai',
    title: '在上海工作后的想法',
    date: '2019-08-20',
    content: `<!-- 文章 HTML 内容 -->`,
    summary: '在上海工作后的感悟...'
  },
  {
    id: 'younan-1',
    title: '囿南记事（篇一）',
    date: '2020-02-03',
    content: `<!-- 文章 HTML 内容 -->`,
    summary: '囿南记事篇一...'
  },
  {
    id: 'qingmingsuibi',
    title: '清明随笔',
    date: '2020-04-04',
    content: `<!-- 文章 HTML 内容 -->`,
    summary: '清明随笔...'
  },
  {
    id: 'noscience2',
    title: '没有科学的世界 (二）',
    date: '2026-05-03',
    content: `<!-- 文章 HTML 内容 -->`,
    summary: '第二章简介...'
  },
  {
    id: 'cpp-smart-pointer',
    title: 'C++ Smart Pointer',
    date: '2020-02-20',
    content: `<!-- 文章 HTML 内容 -->`,
    summary: 'C++ 智能指针笔记...'
  }
]
```

> 未来优化方向：将文章内容存为独立 `.md` 文件，构建时通过 Vite 插件自动导入转换。

---

### 数据迁移对照表

从旧项目迁移内容到新 Vue 组件：

| 旧文件 | 内容 | 迁移目标 |
|-------|------|---------|
| `index.html` 导航栏 | `<div class="navbar">...</div>` | `src/components/Header.vue` |
| `index.html` 页脚 | `<div class="footer-wrap">...</div>` | `src/components/Footer.vue` |
| `index.html` 侧边栏 | `<div class="article-author-side">...</div>` | `src/components/Sidebar.vue` |
| `index.html` 主体内容 | `<article>...</article>` | `src/views/Home.vue` |
| `html/project.html` | 项目内容 | `src/views/Project.vue` |
| `html/blog.html` | 博客列表 | `src/views/Blog.vue` |
| `html/bio.html` | 个人传记 | `src/views/Bio.vue` |
| `html/share.html` | 分享内容 | `src/views/Share.vue` |
| `html/studys/TechBook.html` | 技术书籍 | `src/views/TechBook.vue` |
| `html/blogs/*.html` | 各博客文章 | `src/data/posts.js` + `src/views/BlogPost.vue` |
| `html/Group_3_Project_Website/` | IoT 项目 | `src/views/GroupProject.vue` |

---

### 静态文件处理

以下文件无需编译，直接放入 `vue-app/public/` 目录：

```
vue-app/public/
├── file/
│   ├── Qi Wang.pdf
│   ├── 王琦 简历.pdf
│   └── bigdata.pdf
├── img/
│   └── title-icon.png         # favicon
└── 404.html                    # 旧链接重定向
```

> M6 提升到根目录后，`vue-app/public/` 变为 `public/`，路径不变。

**图片资源决策树：**

- 被 `import` 引用的图片 → 放 `src/assets/images/`（会经过 Vite 哈希化处理）
- 通过固定 URL 直接访问的图片 → 放 `public/img/`
- PDF、favicon 等不经过构建的文件 → 放 `public/`

---

### 部署方案

部署分两个阶段，M5 使用 `vue-migration` 分支 + 子目录构建，M6 合并到 `main` 后切换为标准根目录构建。

#### 阶段一：M5 部署（vue-migration 分支）

详见 M5 里程碑。核心配置：

- GitHub Actions 触发分支：`vue-migration`
- `working-directory: vue-app`
- `cache-dependency-path: vue-app/package-lock.json`
- 构建产物输出到 `./dist`（由 `vite.config.js` 的 `outDir: '../dist'` 控制）

#### 阶段二：M6 合并后（main 分支）

M6 提升到根目录后，GitHub Actions 配置简化：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

> 与 M5 的区别：去掉了 `working-directory` 和 `cache-dependency-path`，触发分支改为 `main`。

---

### `vue-app/public/404.html` - 旧链接重定向

使用 Hash 路由模式后，404.html 主要用于**旧链接的自动映射**。

当用户访问旧的博客文章 URL（如 `/html/blogs/noscience.html`）时，GitHub Pages 会返回 404，触发此页面执行重定向脚本。

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qi's Site</title>
  <script>
    // 旧博客文章 URL 映射到 Hash 路由
    var oldUrlToNewHash = {
      '/html/blogs/noscience.html':   '/#/blog/noscience',
      '/html/blogs/shanghai.html':    '/#/blog/shanghai',
      '/html/blogs/younan-1.html':    '/#/blog/younan-1',
      '/html/blogs/qingmingsuibi.html':'/#/blog/qingmingsuibi',
      '/html/blogs/noscience2.html':  '/#/blog/noscience2',
      '/html/blogs/C++_Smart_Pointer.html': '/#/blog/cpp-smart-pointer',
    };
    var path = location.pathname;
    if (oldUrlToNewHash[path]) {
      location.replace(location.origin + oldUrlToNewHash[path]);
    } else {
      // 非旧文章链接，返回首页
      location.replace(location.origin + '/#/');
    }
  </script>
</head>
<body></body>
</html>
```

> 注意：由于使用了 Hash 路由，`createWebHashHistory` 会自动处理 SPA 内部路由，无需在 `index.html` 中额外处理 `sessionStorage.redirect`。

---

### `package.json` 脚本

开发阶段（M0-M5），deploy 脚本指向 `../dist`：

```json
{
  "name": "qi-website",
  "private": true,
  "version": "2.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.5.0",
    "vue-router": "^4.4.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.1.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "vite": "^6.0.0"
  }
}
```

> M6 提升到根目录后，所有路径中的 `../dist` 相应改为 `dist`。

---

### CSS 迁移策略

| 旧项目 | 新项目 |
|-------|--------|
| `css/main.min.css` | Tailwind 工具类替代大部分样式 |
| `css/academicons.css` | 保留到 `src/assets/styles/`，按需引入 |
| 内联样式 | Tailwind 原子类 |
| Google Fonts (`fonts.googleapis.com`) | 保留在 `index.html` `<head>` 中 |

`src/assets/styles/main.css`：

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* 兼容旧项目中的自定义样式 */
.nav-link {
  @apply text-gray-600 hover:text-gray-900 transition-colors;
}

.blog-title {
  @apply flex items-center gap-4 py-2 border-b border-gray-100;
}
```

---

### 渐进式迁移策略

**核心原则**：在 `vue-migration` 分支上开发，`main` 分支上的旧站始终可访问。Vue 应用在 `vue-app/` 子目录中独立构建，完成所有迁移后一次性替换上线。

```
main 分支（旧站）        vue-migration 分支
┌─────────────────┐     ┌──────────────────────────────┐
│ index.html      │     │ index.html        (旧，不动) │
│ html/blog.html  │     │ html/blog.html    (旧，不动) │
│ html/project.html│     │ html/project.html (旧，不动) │
│ css/            │     │ css/              (旧，不动) │
│ js/             │     │ js/               (旧，不动) │
│                 │     │ vue-app/          (新，开发中)│
│   ↑ 线上运行     │     │   src/components/            │
│   用户访问这个    │     │   src/views/                 │
└─────────────────┘     │   package.json               │
                        │   vite.config.js             │
                        └──────────────────────────────┘
```

分支策略：

```bash
# 创建迁移分支
git checkout -b vue-migration

# 开发期间旧站仍在 main 正常运行
# 需要紧急修复旧站？切回 main 即可
git checkout main
# 修复后
git checkout vue-migration
# 将修复合并过来
git merge main
```

---

### 里程碑计划

#### M0：脚手架搭建

**目标**：在 `vue-app/` 子目录中跑起 Vue 开发服务器，显示占位页。

**不触碰的文件**：所有旧 HTML、CSS、JS 文件保持不动。

```bash
# 创建子目录并初始化
mkdir vue-app
cd vue-app
npm create vite@latest . -- --template vue
npm install
npm install tailwindcss@3 postcss autoprefixer vue-router@4
npx tailwindcss init -p
```

在 `vue-app/` 中创建以下文件：

`vite.config.js`：

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  base: '/',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true
  }
})
```

> `outDir: '../dist'` 将构建产物输出到 `blog/dist/`，不影响旧站点文件。

`src/App.vue`（临时占位）：

```vue
<template>
  <div class="min-h-screen flex items-center justify-center">
    <div class="text-center">
      <h1 class="text-3xl font-bold">Qi Wang's Site - Under Rebuild</h1>
      <p class="text-gray-500 mt-4">Vue3 + Vite + Tailwind CSS</p>
    </div>
  </div>
</template>
```

`tailwind.config.js`、`postcss.config.js`、`src/main.js`、`src/assets/styles/main.css` 按上方"关键文件配置"章节创建。

`vue-app/.gitignore`：

```gitignore
node_modules/
dist/
.DS_Store
*.log
```

**验证方式**：

```bash
cd vue-app
npm run dev
# 浏览器访问 http://localhost:5173，看到 "Under Rebuild" 占位页
```

**Milestone 标准**：

- [x] `npm run dev` 正常启动
- [x] 占位页可访问
- [x] Tailwind CSS 生效（`text-gray-500` 等类名正常渲染）
- [x] 旧站不受任何影响

---

#### M1：布局框架 + 导航栏

**目标**：实现完整的页面布局（Header、Footer、Sidebar），导航栏所有链接可点击，能跳转到占位页面。

**新增文件**：

- `vue-app/src/components/Header.vue`（按上方组件设计）
- `vue-app/src/components/Footer.vue`
- `vue-app/src/components/Sidebar.vue`
- `vue-app/src/components/Dropdown.vue`
- `vue-app/src/router/index.js`（所有路由指向占位组件）

**修改文件**：

- `vue-app/src/App.vue`（从占位页改为使用布局组件 + `<router-view>`）
- `vue-app/src/main.js`（引入 router）

**占位页面**（每个 views 文件先用简单模板）：

```vue
<!-- vue-app/src/views/Home.vue 等所有页面 -->
<template>
  <div>
    <h1 class="text-2xl font-bold">{{ $route.name }}</h1>
    <p class="text-gray-500 mt-2">Page under construction</p>
  </div>
</template>
```

**验证方式**：

```bash
cd vue-app
npm run dev
# 浏览器访问 localhost:5173，看到完整布局（导航栏 + 侧边栏 + 页脚）
# 点击导航栏各链接，页面内容区域切换
# 下拉菜单（MY STORIES / CV）鼠标悬停展开
```

**Milestone 标准**：

- [x] Header 导航栏完整显示
- [x] 下拉菜单正常展开和收起
- [x] Sidebar 显示头像和社交链接
- [x] Footer 显示版权信息
- [x] 点击导航链接，`<router-view>` 区域切换
- [x] 所有路由可达，无控制台报错

---

#### M2：逐页内容迁移

**目标**：将旧 HTML 页面内容逐一迁移到 Vue 组件，每迁移一个页面就验证一个。

**迁移顺序**（按重要性从高到低）：

| 步骤 | 旧文件 | Vue 组件 | 验证点 |
|-----|-------|---------|-------|
| 2a | `index.html` 主体内容 | `vue-app/src/views/Home.vue` | 个人介绍段落、机器人项目描述完整 |
| 2b | `html/project.html` | `vue-app/src/views/Project.vue` | Tiling Robot、IoT 项目、YouTube 视频嵌入 |
| 2c | `html/bio.html` | `vue-app/src/views/Bio.vue` | 传记内容完整 |
| 2d | `html/share.html` | `vue-app/src/views/Share.vue` | 分享内容完整 |
| 2e | `html/studys/TechBook.html` | `vue-app/src/views/TechBook.vue` | 技术书籍列表 |
| 2f | `html/Group_3_Project_Website/index.html` | `vue-app/src/views/GroupProject.vue` | IoT 项目详情、图片和 iframe |

**每个子步骤的流程**：

```
1. 读取旧 HTML 文件中的 <article> 或主体内容
   ↓
2. 将内容复制到对应 Vue 组件，用 Tailwind 类名替换旧 CSS 类
   ↓
3. 将旧 HTML 中引用的图片迁移到 vue-app/src/assets/images/
   ↓
4. npm run dev，浏览器验证该页面内容与旧站一致
   ↓
5. 提交代码：git commit -m "M2a: migrate Home page"
```

**迁移注意事项**：

- YouTube `<iframe>` 嵌入代码直接保留原样
- LaTeX 公式图片链接（`latex.codecogs.com`）直接保留原样
- 外部链接（GitHub、LinkedIn 等）保持不变
- 图片路径改为 `@/assets/images/xxx` 或相对路径

**验证方式**：每个子步骤完成后，打开新站对应页面，与旧站页面**并排对比**，确认内容一致。

**Milestone 标准**：

- [x] 所有 6 个页面内容与旧站一致
- [x] 图片正常显示
- [x] 外部链接可点击
- [x] YouTube 视频可播放
- [x] 在 `vue-app/` 目录运行 `npm run dev`，所有页面无报错

---

#### M3：博客系统

**目标**：实现数据驱动的博客列表和文章详情页，替代旧的 6 个独立 HTML 文件。

**新增文件**：

- `vue-app/src/data/posts.js`（从旧 HTML 提取文章内容，存为结构化数据）
- `vue-app/src/views/Blog.vue`（博客列表，按日期倒序排列）
- `vue-app/src/views/BlogPost.vue`（文章详情，通过 `useRoute().params.id` 加载）
- `vue-app/src/components/BlogCard.vue`（列表卡片）

**数据结构**（`posts.js`）：

```javascript
export const posts = [
  {
    id: 'noscience',
    title: '没有科学的世界 (一）',
    date: '2018-07-11',
    summary: '当科学从世界上消失...',
    content: `
      <p>假设明天醒来，世界上所有的科学知识...</p>
      <!-- 完整文章 HTML 内容 -->
    `
  },
  // ... 其他文章
]
```

**迁移步骤**：

```
1. 从 html/blogs/noscience.html 提取文章内容 → posts.js[0]
2. 从 html/blogs/shanghai.html 提取文章内容 → posts.js[1]
3. 从 html/blogs/younan-1.html 提取文章内容 → posts.js[2]
4. 从 html/blogs/qingmingsuibi.html 提取文章内容 → posts.js[3]
5. 从 html/blogs/noscience2.html 提取文章内容 → posts.js[4]
6. 从 html/blogs/C++_Smart_Pointer.html 提取文章内容 → posts.js[5]
7. 实现 Blog.vue 列表页
8. 实现 BlogPost.vue 详情页
9. 逐篇对比新旧文章内容
```

**验证方式**：

```bash
cd vue-app
npm run dev
# 访问 /blog，看到 6 篇文章列表（按日期倒序）
# 点击每篇文章，内容与旧站一致
# 直接访问 /blog/noscience，文章正常加载（SPA 路由）
```

**Milestone 标准**：

- [x] 博客列表按日期倒序排列
- [x] 点击文章标题进入详情页
- [x] 6 篇文章内容与旧站逐字对比一致
- [x] 路由参数 `:id` 正确匹配文章

---

#### M4：静态资源 + 构建验证

**目标**：处理静态文件（PDF、图片），验证 `npm run build` 产出的 `dist/` 目录可独立运行。

**迁移静态文件**：

```
vue-app/
├── public/
│   ├── file/
│   │   ├── Qi Wang.pdf          ← 从 blog/file/ 复制
│   │   ├── 王琦 简历.pdf         ← 从 blog/file/ 复制
│   │   └── bigdata.pdf          ← 从 blog/file/ 复制
│   └── img/
│       └── title-icon.png       ← 从 blog/img/ 复制
```

> **CNAME**：如果仓库使用了自定义域名，需将 `CNAME` 文件复制到 `vue-app/public/CNAME`，否则部署后域名配置会丢失。
>
> **feed.xml**：旧站的 `js/feed.xml`（RSS 订阅文件）位于 `js/` 目录下。如需保留 RSS 功能，将其复制到 `vue-app/public/feed.xml`，后续可考虑构建时自动生成。
>
> **photo/roboticplus.jpeg**：旧站 `photo/` 目录中的 `roboticplus.jpeg` 未在任何页面中被引用，M4 无需迁移。M6 清理旧文件时一并删除。
>
> **academicons.css**：旧站加载了学术图标库 `css/academicons.css`，但当前页面中未使用到 academicons 图标，迁移时可忽略。如有需要，后续可通过 npm 安装 `academicons` 包。
>
> **开发阶段文件共存**：M0-M5 期间，旧站的 `blog/index.html` 和 Vue 的 `vue-app/index.html` 各自独立。`npm run dev` 在 `vue-app/` 内运行（端口 5173），`npm run build` 输出到 `blog/dist/`，不会覆盖旧站的 `blog/index.html`。旧站继续通过 GitHub Pages 的 `main` 分支正常运行。

**验证构建**：

```bash
cd vue-app
npm run build

# 构建产物在 blog/dist/
# 用预览服务器验证
npx vite preview

# 或用 Python 简单起一个静态服务
cd ../dist
python3 -m http.server 4173
# 浏览器访问 http://localhost:4173
```

**构建验证清单**：

- [x] `dist/` 目录生成成功
- [x] 所有页面可访问
- [x] PDF 文件可下载（`/file/Qi Wang.pdf`）
- [x] 图片正常显示
- [x] 浏览器控制台无报错
- [x] `dist/` 文件大小合理（预期 < 500KB，不含 PDF）

---

#### M5：部署切换

**目标**：将 Vue 站点部署到 GitHub Pages，替换旧站。

**步骤一：配置 GitHub Actions**

在 `vue-migration` 分支创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [vue-migration]

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: vue-app/package-lock.json

      - name: Install dependencies
        working-directory: vue-app
        run: npm ci

      - name: Build
        working-directory: vue-app
        run: npm run build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

> 注意 `working-directory: vue-app`，确保在子目录中执行构建。

**步骤二：创建旧链接重定向页面**

在 `vue-app/public/404.html` 中创建（按上方"旧链接重定向"章节配置）。

**步骤三：推送并验证**

```bash
# 在 vue-migration 分支
git add .
git commit -m "M5: ready for deployment"
git push origin vue-migration

# GitHub Actions 自动构建并部署到 gh-pages 分支
# 在 GitHub 仓库 Settings → Pages → Source 选择 gh-pages 分支
```

**步骤四：上线后验证**

| 验证项 | URL | 预期结果 |
|-------|-----|---------|
| 首页 | `qw2261.github.io/` | 个人介绍页 |
| 博客列表 | `qw2261.github.io/#/blog` | 6 篇文章列表 |
| 博客详情 | `qw2261.github.io/#/blog/noscience` | 文章内容 |
| 项目页 | `qw2261.github.io/#/project` | 项目展示 |
| PDF 下载 | `qw2261.github.io/file/Qi Wang.pdf` | PDF 文件 |
| 旧链接重定向 | `qw2261.github.io/html/blogs/noscience.html` | 自动跳转到 `/#/blog/noscience` |

**Milestone 标准**：

- [x] GitHub Actions 构建成功
- [x] gh-pages 分支已更新
- [x] 所有页面在线可访问
- [x] 旧链接重定向正常（访问 `/html/blogs/noscience.html` 自动跳转到 `/#/blog/noscience`）

---

#### M6：收尾清理 + 提升到根目录

**目标**：将 `vue-app/` 内容提升到根目录，删除所有旧文件，更新构建配置，合并分支。

**步骤一：合并 vue-migration 到 main**

```bash
git checkout main
git merge vue-migration
```

**步骤二：将 vue-app/ 内容提升到根目录**

```bash
# 移动 Vue 应用文件到根目录
# 注意：mv 会覆盖根目录同名文件（如 index.html），这正是我们需要的
mv vue-app/src ./
mv vue-app/public ./
mv vue-app/index.html ./
mv vue-app/vite.config.js ./
mv vue-app/tailwind.config.js ./
mv vue-app/postcss.config.js ./
mv vue-app/package.json ./
mv vue-app/package-lock.json ./

# 删除空的 vue-app 目录
rm -rf vue-app/
```

**步骤三：更新 vite.config.js**

```bash
# 将 outDir 从 '../dist' 改为 'dist'
sed -i '' "s/outDir: '..\/dist'/outDir: 'dist'/" vite.config.js
```

**步骤四：更新 package.json**

```bash
# 移除 gh-pages 包和 deploy 脚本（改用 GitHub Actions 自动部署）
sed -i '' '/"deploy":/d' package.json
npm uninstall gh-pages
```

**步骤五：更新 GitHub Actions**

将 `.github/workflows/deploy.yml` 中的 M5 配置替换为 M6 配置（见"部署方案 → 阶段二"章节）：
- 触发分支从 `vue-migration` 改为 `main`
- 去掉 `working-directory: vue-app`
- 去掉 `cache-dependency-path: vue-app/package-lock.json`

**步骤六：删除旧文件**

```bash
git rm -r html/
git rm -r css/
git rm -r js/
git rm -r fonts/
git rm -r photo/
git rm -r .jekyll-cache/
git rm _config.yml
```

> 注意：旧站 `index.html` 已在步骤二被 Vue 的 `index.html` 覆盖，无需单独删除。

**步骤七：更新 .gitignore**

```gitignore
node_modules/
dist/
.DS_Store
*.log
.vscode/
*.swp
*.swo
.jekyll-cache/
.jekyll-metadata
```

**步骤八：最终验证**

```bash
npm install
npm run build
npx vite preview
# 浏览器验证所有页面正常

git add .
git commit -m "M6: elevate to root, cleanup old files"
git push origin main
# 访问 qw2261.github.io，确认站点正常
```

**Milestone 标准**：

- [x] `vue-app/` 目录已删除，所有文件在根目录
- [x] `vite.config.js` 的 `outDir` 为 `'dist'`
- [x] GitHub Actions 触发分支为 `main`，无 `working-directory`
- [x] 所有旧文件已删除
- [x] `npm run build` 成功，站点在线正常运行
- [x] `vue-migration` 分支可以删除

---

### 里程碑总览

```
M0 脚手架搭建      →  Vue 开发服务器跑起来，显示占位页
                     ↓
M1 布局 + 导航      →  Header / Footer / Sidebar / 路由跳转
                     ↓
M2 逐页迁移内容     →  Home → Project → Bio → Share → TechBook → GroupProject
                     ↓
M3 博客系统         →  数据驱动博客列表 + 文章详情
                     ↓
M4 构建验证         →  npm run build 产出 dist/，独立可运行
                     ↓
M5 部署上线         →  GitHub Actions 部署，旧站替换为新站
                     ↓
M6 清理收尾         →  提升到根目录，删除旧文件，合并分支
```

**每个 Milestone 的安全性**：

| 特性 | 说明 |
|-----|------|
| **旧站在线** | M0-M4 期间，main 分支旧站始终正常运行 |
| **独立分支** | 所有开发在 `vue-migration` 分支，不触碰 main |
| **可回滚** | 任何阶段可切回 main，旧站 100% 恢复 |
| **可验证** | 每个 Milestone 有明确的验证清单 |
| **逐页迁移** | M2 阶段每迁移一个页面就验证一个，不会一次性全崩 |

**回滚操作步骤**：

```bash
# 场景1：M3 开发分支出问题，需要重建
git checkout main                        # 切换到旧站（完全正常）
git branch -D vue-migration              # 删除有问题的分支
git checkout -b vue-migration            # 重新创建干净分支
# 重新从 M0 开始

# 场景2：M5 部署后发现线上问题
# 方案A：GitHub Pages 设置回退
# 仓库 Settings → Pages → Source 临时切回 main 分支（或任意包含旧站文件的分支）
# GitHub 会自动从该分支的根目录 serve，旧的 index.html 立即生效

# 方案B：使用 git revert 撤销 M5 的部署提交
git checkout vue-migration
git log                                  # 找到 M5 部署提交 hash
git revert <commit-hash>                # 撤销该提交
git push origin vue-migration           # 触发新的构建部署

# 场景3：M6 合并后发现严重问题
# 在本地或 GitHub 上将 main 回退到合并前的提交
git checkout main
git reset --hard <merge前的commit-hash>
git push origin main --force            # 危险操作，需谨慎
```

---

### 风险与注意事项

| 风险 | 应对措施 |
|-----|---------|
| SEO 影响 | Vue SPA 天生 SEO 不友好（无 SSR）。如需改善，可使用 `@unhead/vue` 管理 meta 信息，但无法根本解决 |
| SPA 路由 404 | 使用 `createWebHashHistory`（推荐），GitHub Pages 原生支持，无需额外配置 |
| 旧链接失效 | `vue-app/public/404.html` 中包含旧 URL → Hash 路由映射脚本 |
| 构建体积过大 | Vue Router 使用懒加载（`() => import(...)`），图片使用 Vite 压缩 |
| GitHub Pages 缓存 | 部署后如果页面没更新，等待几分钟或强制刷新（Ctrl+Shift+R） |
| 子目录构建路径 | M0-M5：`vite.config.js` 中 `outDir: '../dist'`，GitHub Actions 中 `working-directory: vue-app`；M6 后统一为根目录 |
| RSS Feed | 旧站 `js/feed.xml` 在 M4 阶段迁移到 `vue-app/public/feed.xml`，M6 后变为 `public/feed.xml` |
| Google Analytics | 如旧站使用了 GA，需将跟踪代码迁移到 `App.vue` 的 `onMounted` 中 |
| CNAME 丢失 | 如使用自定义域名，`CNAME` 文件需放入 `vue-app/public/CNAME`（M4），M6 后变为 `public/CNAME` |
| M6 提升操作风险 | `mv` 操作前确保已 commit，如有问题可 `git checkout .` 恢复 |

### 待确认事项

| 项目 | 说明 |
|-----|------|
| **Google Analytics** | 旧站是否使用 Google Analytics 或其他统计工具？ |
| **CNAME** | 仓库是否使用自定义域名？如有，需在 M4 将 `CNAME` 放入 `vue-app/public/` |

---

### 旧文件处理

M6 清理阶段删除：

- `html/` - 所有旧 HTML 页面
- `css/` - 旧样式（被 Tailwind 替代）
- `js/` - 旧脚本（jQuery、Modernizr、scripts.min.js、feed.xml）
- `fonts/` - 旧字体（由 Google Fonts CDN 提供）
- `photo/` - 旧照片（已迁移到 `src/assets/images/`）
- `_config.yml` - Jekyll 配置
- `.jekyll-cache/` - Jekyll 缓存

> 旧站 `index.html` 在 M6 步骤二被 Vue 的 `index.html` 覆盖，无需单独删除。

迁移后保留的文件：

- `img/` 中的图片 → 迁移到 `src/assets/images/` 或 `public/img/`
- `file/*.pdf` → 迁移到 `public/file/`
- `js/feed.xml` → 迁移到 `public/feed.xml`（如需保留 RSS）
- `CNAME`（如果有自定义域名）→ 迁移到 `public/CNAME`
