<template>
  <div>
    <h1 class="text-2xl font-bold text-gray-900">Smart Movable Trash-Bin With Trash Identification in Office</h1>
    <p class="text-sm text-gray-500 mt-1">Columbia University · EECS E4764 Fall'19 · Team 3 Project Report</p>
    <div class="mt-6 space-y-8 text-gray-700">

      <section id="abstract">
        <div class="relative w-full" style="padding-bottom: 56.25%; height: 0;">
          <iframe src="https://player.bilibili.com/player.html?aid=79476269&cid=136003950&page=1"
            class="absolute top-0 left-0 w-full h-full" frameborder="0" allowfullscreen></iframe>
        </div>
        <h2 class="text-xl font-semibold mt-4">Abstract</h2>
        <p class="mt-2">
          Smart Movable Trash-Bin (SMTB) is an IoT solution to current trash sorting problem in office.
          Currently, the trash sorting is a trend and a remedial action to save the environment, which
          becomes a policy of great significance in numerous countries and cities. However, people, especially
          the officer in office, who are not going to have a single trash-bin suffers
          from the trash sorting and often feel confused about the trash type every time they are going to
          drop trash. Our project aims to automate
          the process in the origin, which means when people drop trash, the trash is accurately identifed as
          the trash type, including metal, plastic, glass, paper, cardboard (recyclable in most regions)
          and trash (unrecyclable). In addition, our target is to make it as a intelligent project.
          Since in most cases, officers should share a common trash-bin and need to move when they have trash.
          But with our design, user can directly use phone to call our SMTB to come to his/her postion and user
          can simply put trash in and SMTB intelligently identify the trash type and go back orginal place for
          sorting and collecting.
        </p>
        <p class="mt-2">
          With SMTB, officers will benefit from simplification of dropping trash and trash sorting; office
          will become more clean; government will benefit from the energy conservation resulting from resource
          reuse. We hope our project will make a difference to the world.
        </p>
      </section>

      <section id="motivation">
        <h2 class="text-xl font-semibold border-b border-gray-200 pb-1">Motivation</h2>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="https://www.telegraph.co.uk/content/dam/news/peoples-daily/chinese-woman-emptying-bin.jpg"
            alt="Trash sorting" class="w-full" />
        </div>
        <p class="mt-2">
          2019 has witnessed the stablishment of compulsary trash-sorting policy in many major
          cities in China and USA. Our self-trash-sorting trash-bin will help people figure out the correct
          sort of trash. It will save people's time and also avoid the incorrect-sorting case.
        </p>
        <div class="mt-2 max-w-lg mx-auto">
          <img
            src="https://www.simplehuman.com/media/wysiwyg/feature-chapter/rectangular-sensor-can2-sensor-D.jpg"
            alt="Smart trash bin" class="w-full" />
        </div>
        <p class="mt-2">
          Smart house and office products like smart sweepers have become popular theses days.
          While most smart trash-bin products in market fail to provide user-friendly and fully automative
          service but only with functions of great confinement such as auto-opening using voice or movition sensor,
          our project is going to provide a new perspective.
        </p>
        <p class="mt-2">
          Our motivation fully automating the trash collecting and sorting stimulates us to create SMTB.
        </p>
        <p class="mt-2">
          The SMTB designed by us still resembles the use logic of theses product to make all functions
          accessible from phone. However, we manage to make the trash collecting and trash sorting automated in SMTB.
          When user use the phone to call SMTB, it will immediately approach users once there is no other
          unfinished command. Then it will collect the trash and identify the trash type. At the same time, SMTB will
          come back to the original place for battery charging, trash collection and waiting for the next command.
          We hope SMTB provides user-friendly enough service for users.
        </p>
      </section>

      <section id="system">
        <h2 class="text-xl font-semibold border-b border-gray-200 pb-1">System</h2>

        <h3 class="text-lg font-semibold mt-4">Architecture</h3>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/architecture.png" alt="Architecture" class="w-full" />
        </div>

        <h3 class="text-lg font-semibold mt-6">Technical Components</h3>
        <p class="mt-2">There are three main technical components:</p>

        <h4 class="text-base font-semibold mt-4">APP Design</h4>
        <div class="flex flex-wrap gap-4 mt-2">
          <div class="text-center">
            <img src="/img/group/app1.jpeg" class="w-40" alt="App Screenshot 1" />
            <p class="text-sm mt-1">Home page with two buttons</p>
          </div>
          <div class="text-center">
            <img src="/img/group/app2.jpeg" class="w-40" alt="App Screenshot 2" />
            <p class="text-sm mt-1">Take photo to identify trash type</p>
          </div>
          <div class="text-center">
            <img src="/img/group/app3.jpeg" class="w-40" alt="App Screenshot 3" />
            <p class="text-sm mt-1">Call trash can</p>
          </div>
        </div>
        <p class="mt-2">
          We used Android Studio to develop a phone APP. There are two functions along with 3 layouts.
          Take photo layout identifies trash type from photo. Call trash can layout sends user position
          to request the SMTB.
        </p>

        <h4 class="text-base font-semibold mt-4">Server Design</h4>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/server.png" alt="Server Architecture" class="w-full" />
        </div>
        <p class="mt-2">
          The server is built by Django with endpoints /recommend, /identify, /acceptCommand and /actCommand.
          Made public via ngrok. The trash identification model uses TensorFlow with Inception-V3 architecture.
        </p>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/ESP8266.png" alt="ESP8266" class="w-full" />
        </div>
        <p class="mt-2">
          ESP8266 board listens to /actCommand every 3 seconds for new user commands, then interacts with STM32
          on the car to execute the instruction. The camera uploads images to server through /identify for trash
          classification.
        </p>

        <h4 class="text-base font-semibold mt-4">Car Design</h4>
        <p class="mt-2">
          The car has two parts: a trash can with OV2640 camera and an omnidirectional chassis.
          Four Mecanum wheels enable omnidirectional movement. Controlled by STM32F407 board with
          localization system using two orthogonal encoders and IMU.
        </p>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/car_design.png" alt="Car Design" class="w-full" />
        </div>

        <h3 class="text-lg font-semibold mt-6">Prototype</h3>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/prototype.jpeg" alt="Prototype" class="w-full" />
        </div>
        <p class="mt-2">The prototype combines a car and trash bin.</p>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/server_prototype.png" alt="Server Prototype" class="w-full" />
        </div>
        <div class="flex flex-wrap gap-4 mt-2">
          <img src="/img/group/app_prototype1.png" class="w-64" alt="App Prototype 1" />
          <img src="/img/group/app_prototype2.png" class="w-64" alt="App Prototype 2" />
        </div>
      </section>

      <section id="results">
        <h2 class="text-xl font-semibold border-b border-gray-200 pb-1">Results</h2>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/result_1.png" alt="Results 1" class="w-full" />
        </div>
        <p class="mt-2">We have completed most parts of our wish list and nearly reached Gold outcome except obstacle
          avoiding.</p>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/result_2.png" alt="Results 2" class="w-full" />
        </div>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/result_4.png" alt="Results 4" class="w-full" />
        </div>
        <div class="mt-2 max-w-lg mx-auto">
          <img src="/img/group/result_3.png" alt="Results 3" class="w-full" />
        </div>
        <p class="mt-2">
          The accuracy of trash identification in training and validation sets reaches up to 87%.
          The car has stable and accurate localization as shown in the demo video.
          The phone can take a photo and display the identified trash type, and allow users to monitor and command the
          SMTB.
        </p>
      </section>

      <section id="references">
        <h2 class="text-xl font-semibold border-b border-gray-200 pb-1">References</h2>
        <ol class="mt-2 list-decimal pl-5 space-y-1 text-sm">
          <li>B. Chowdhury and M. U. Chowdhury, "RFID-based Real-time Smart Waste Management System," in Australasian
            Telecommunication Networks and Applications Conference, 2007, no. December, pp. 175–180.</li>
          <li>A. Zanella, N. Bui, A. Castellani, L. Vangelista, and M. Zorzi, "Internet of Things for Smart Cities,"
            IEEE Internet Things J., vol. 1, no. 1, pp. 22–32, 2014.</li>
          <li>F. Mattern, "From smart devices to smart everyday objects," Proc. Smart Objects Conf., no. April, pp.
            15–16, 2003.</li>
          <li>BigBellySolar, "CNN - Solar Powered Trash Compactors," 2010. [Online]. Available:
            https://www.youtube.com/watch?v=8e8Be9rq_C8.</li>
          <li>S. Zavare, R. Parashare, S. Patil, P. Rathod, and P. V. Babanne, "Smart City Waste Management System
            Using GSM," Int. J. Comput. Sci. Trends Technol., vol. 5, no. 3, pp. 74–78, 2017.</li>
          <li>T. Singh, R. Mahajan, and D. Bagai, "Smart Waste Management using Wireless Sensor Network," Int. J.
            Innov. Res. Comput. Commun. Eng., vol. 4, no. 6, pp. 10343–10347, 2016.</li>
          <li>S. S. Navghane, M. S. Killedar, and V. M. Rohokale, "IoT Based Smart Garbage and Waste Collection Bin,"
            Int. J. Adv. Res. Electron. Commun. Eng., vol. 5, no. 5, pp. 1576–1578, 2016.</li>
          <li>T. Ambrose, C. Ford, and M. Norris, "Smart City Trash Cans," California Polytechnic State University,
            2015.</li>
          <li>Ecube Labs, "Smart Waste Management System | Waste Analytics | Ecube Labs." [Online]. Available:
            http://ecubelabs.com/.</li>
          <li>Bigbelly, "Bigbelly - Smart City Solutions." [Online]. Available: http://bigbelly.com/.</li>
        </ol>
      </section>

      <section id="team">
        <h2 class="text-xl font-semibold border-b border-gray-200 pb-1">Our Team</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
          <div class="text-center">
            <img src="https://qw2261.github.io/img/head.jpeg" alt="Qi Wang" class="w-32 h-32 rounded-full mx-auto" />
            <h3 class="font-semibold mt-2">
              <a href="https://qw2261.github.io/" class="text-blue-600 hover:underline">Qi Wang</a>
            </h3>
            <p class="text-sm mt-1">Master student in Mechanical Engineering (Robotics) at Columbia University.</p>
          </div>
          <div class="text-center">
            <img src="/img/group/yunqingxiao.png" alt="Yunqing Xiao" class="w-32 h-32 rounded-full mx-auto" />
            <h3 class="font-semibold mt-2">
              <a href="https://www.linkedin.com/in/yunqingxiao/" class="text-blue-600 hover:underline">Yunqing Xiao</a>
            </h3>
            <p class="text-sm mt-1">Master student in Computer Engineering at Columbia University.</p>
          </div>
          <div class="text-center">
            <img src="/img/group/zhihaozheng.jpeg" alt="Zhihao Zheng" class="w-32 h-32 rounded-full mx-auto" />
            <h3 class="font-semibold mt-2">Zhihao Zheng</h3>
            <p class="text-sm mt-1">Master student of Computer Engineering at Columbia University. In charge of chassis.
            </p>
          </div>
        </div>
      </section>

      <section id="contact">
        <h2 class="text-xl font-semibold border-b border-gray-200 pb-1">Contact</h2>
        <div class="mt-2 space-y-1 text-sm">
          <p><strong>Qi Wang: </strong><a href="mailto:qw2261@columbia.edu"
              class="text-blue-600 hover:underline">qw2261@columbia.edu</a></p>
          <p><strong>Yunqing Xiao: </strong><a href="mailto:yx2542@columbia.edu"
              class="text-blue-600 hover:underline">yx2542@columbia.edu</a></p>
          <p><strong>Zhihao Zheng: </strong><a href="mailto:zz2698@columbia.edu"
              class="text-blue-600 hover:underline">zz2698@columbia.edu</a></p>
          <p class="mt-2"><a href="https://github.com/qw2261/IoT-Project" class="text-blue-600 hover:underline"><strong>Project
                Code</strong></a> is uploaded to GitHub!</p>
          <p class="mt-2"><strong>Columbia University </strong><a href="http://www.ee.columbia.edu"
              class="text-blue-600 hover:underline">Department of Electrical Engineering</a></p>
          <p><strong>Instructor:</strong> <a href="http://fredjiang.com/" class="text-blue-600 hover:underline">Professor
              Xiaofan (Fred) Jiang</a></p>
        </div>
      </section>

    </div>
  </div>
</template>
