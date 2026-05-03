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
