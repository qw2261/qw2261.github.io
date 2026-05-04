<template>
  <header class="bg-white/95 backdrop-blur-md border-b border-gray-100 sticky top-0 z-50 shadow-sm">
    <div class="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
      <router-link to="/" class="text-lg font-bold text-gray-900 hover:text-gray-600 transition-all duration-300 font-serif">
        Qi Wang
      </router-link>
      <nav class="hidden md:flex items-center space-x-8">
        <router-link to="/" class="nav-link">Home</router-link>
        <router-link to="/project" class="nav-link">Projects</router-link>
        <router-link to="/blog" class="nav-link">Blog</router-link>
        <Dropdown title="MY STORIES" :items="storiesItems" />
        <Dropdown title="CV" :items="cvItems" />
      </nav>
      <button 
        class="md:hidden text-gray-600 hover:text-gray-900 transition-all duration-300 hover:scale-110 active:scale-95" 
        @click="mobileOpen = !mobileOpen"
      >
        <svg class="w-6 h-6 transition-transform duration-300" :class="{ 'rotate-90': mobileOpen }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
    </div>
    <Transition name="slide">
      <div v-if="mobileOpen" class="md:hidden bg-white/98 backdrop-blur-md border-t border-gray-100 shadow-lg">
        <div class="px-6 py-4 space-y-1">
          <button 
            @click="handleNavClick('/')" 
            class="w-full text-left nav-link py-3 px-4 rounded-lg hover:bg-gray-50 transition-all duration-300 hover:translate-x-1 active:scale-[0.98]"
          >Home</button>
          <button 
            @click="handleNavClick('/project')" 
            class="w-full text-left nav-link py-3 px-4 rounded-lg hover:bg-gray-50 transition-all duration-300 hover:translate-x-1 active:scale-[0.98]"
          >Projects</button>
          <button 
            @click="handleNavClick('/blog')" 
            class="w-full text-left nav-link py-3 px-4 rounded-lg hover:bg-gray-50 transition-all duration-300 hover:translate-x-1 active:scale-[0.98]"
          >Blog</button>
          <div class="py-3">
            <div class="nav-link font-semibold text-gray-700 px-4">MY STORIES</div>
            <div class="pl-4 mt-2 space-y-1">
              <button 
                v-for="(item, index) in storiesItems" 
                :key="item.label" 
                @click="item.to ? handleNavClick(item.to) : window.open(item.href, item.external ? '_blank' : '_self')" 
                class="w-full text-left text-sm text-gray-600 py-2.5 px-4 rounded-lg hover:text-gray-900 hover:bg-gray-50 transition-all duration-300 hover:translate-x-1 active:scale-[0.98]"
                :style="{ animationDelay: `${index * 50}ms` }"
              >{{ item.label }}</button>
            </div>
          </div>
          <div class="py-3">
            <div class="nav-link font-semibold text-gray-700 px-4">CV</div>
            <div class="pl-4 mt-2 space-y-1">
              <button 
                @click="window.open('/file/Qi Wang.pdf', '_blank')" 
                class="w-full text-left text-sm text-gray-600 py-2.5 px-4 rounded-lg hover:text-gray-900 hover:bg-gray-50 transition-all duration-300 hover:translate-x-1 active:scale-[0.98]"
              >Qi Wang's CV</button>
              <button 
                @click="window.open('/file/王琦 简历.pdf', '_blank')" 
                class="w-full text-left text-sm text-gray-600 py-2.5 px-4 rounded-lg hover:text-gray-900 hover:bg-gray-50 transition-all duration-300 hover:translate-x-1 active:scale-[0.98]"
              >王琦 简历</button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </header>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import Dropdown from './Dropdown.vue'
import { storiesItems, cvItems } from '@/data/navigation'

const mobileOpen = ref(false)
const router = useRouter()

const handleNavClick = (path) => {
  mobileOpen.value = false
  router.push(path)
}
</script>

<style scoped>
.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

button:active {
  transition: transform 0.1s ease-out;
}
</style>
