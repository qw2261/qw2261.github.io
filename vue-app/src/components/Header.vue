<template>
  <header class="bg-white border-b border-gray-200 sticky top-0 z-50">
    <div class="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
      <router-link to="/" class="text-xl font-bold text-gray-800 hover:text-gray-600">
        Qi Wang's site
      </router-link>
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
    <div v-if="mobileOpen" class="md:hidden bg-white border-t px-4 py-3 space-y-1">
      <router-link to="/" class="block nav-link py-2" @click="mobileOpen = false">Home</router-link>
      <router-link to="/project" class="block nav-link py-2" @click="mobileOpen = false">Projects</router-link>
      <router-link to="/blog" class="block nav-link py-2" @click="mobileOpen = false">Blog</router-link>
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
