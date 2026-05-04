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
    path: '/algorithm-training',
    name: 'AlgorithmTraining',
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
  history: createWebHashHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  }
})

export default router
