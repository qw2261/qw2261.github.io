<template>
  <div>
    <h1 class="text-2xl font-bold text-gray-900">Selected Projects</h1>
    <div class="mt-4 space-y-6 text-gray-700">

      <section>
        <h3 id="tiling_robot" class="text-lg font-semibold">Tiling Robot Design</h3>
        <p class="mt-2">
          In this project, I participated in a tiling robot for room and hall.
          I am mainly responsible for Aubo Force Feedback System and Mortar Injection process.
          To build Force Feedback System, I attempted to resort to torque sensor but the weight
          of sensor seriously affect the working efficiency of tiling. So, a simpler but reliable
          method is necessary.
        </p>

        <h4 class="text-base font-semibold mt-4">1. Aubo Force Feedback System</h4>
        <p class="mt-2">
          I visited <a href="https://github.com/lg609/aubo_robot" class="text-blue-600 hover:underline">Aubo Official
            SDK</a> and
          collected electrical current of every joint. From
          <a href="https://journals.sagepub.com/doi/full/10.1177/1729881419846712" class="text-blue-600 hover:underline">a
            paper</a>
          , there should be a linear regression model for currents and joint torques. In this way, I used
          torque sensor to collect torques and corresponding currents. After that, I trained a linear
          regress model to predict torques by currents.
        </p>
        <p class="mt-2">
          Then, based on equation
          <img
            src="https://latex.codecogs.com/gif.latex?\inline&space;J^T[F_x,&space;F_y,&space;F_z,&space;\tau_x,&space;\tau_y,&space;\tau_z]&space;=&space;[\tau_1,&space;\tau_2,&space;\tau_3,&space;\tau_4,&space;\tau_5,&space;\tau_6]"
            class="inline-block align-middle my-2" />,
          I used <a href="https://moveit.ros.org/" class="text-blue-600 hover:underline">MoveIt!</a> to obtain the joint
          states and Jacobian
          matrix based upon URDF of Aubo Robot to compute the force and torque in the end-effector with the help of
          Eigen
          in C++ for matrix evaluation. The computed force and torque in the end-effector are published in a /Wrench
          topic
          as WrenchStamp msg with a frequency of 100 hz.
        </p>
        <p class="mt-2">
          Then, I put forward a force feedback system. The principle is that the laser in the end-effector
          will keep inspecting the distance between the end-effector and object and reminds system to mark
          current force as force threshold when the distance becomes smaller than the distance threshold which
          is set according to experience in various working environments. Then, as the end-effector constantly closes
          to object, the system will examine whether the force difference between current force and force threshold
          is larger than the set force_difference_threshold. Once the system confirms the result that force difference
          is too large, the state of robot will be updated as touched and follow up a series of operations according to
          the robot task.
        </p>

        <h4 class="text-base font-semibold mt-4">2. Mortar Injection</h4>
        <p class="mt-2">
          After completion of Force Feedback System, I am assigned a project to finish the mortar Injection
          process design. In my design, the robot arm will initially to move to a position to scan the ARcode
          by camera to recoginize the accurate position of the board. Then the robot will plan to suck the board
          with the help of the Aubo Force Feedback System. The detail of operation is shown in the video.
        </p>
        <div class="relative w-full mt-2" style="padding-bottom: 56.25%; height: 0;">
          <iframe src="https://www.youtube.com/embed/7tuRClAwbnk"
            class="absolute top-0 left-0 w-full h-full" frameborder="0"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        <p class="mt-2">
          Once sucking the board, the robot arm with board will move to initial position to start mortar Injection.
          Based on reacheable range of robot arm and the reason that we are not willing to inject the mortar on the
          surface of tile
          to lower the cleanness, the robot will autonomously adjust the injecting range (with and length) according to
          the position of
          corner of tile of last unit in the base link coordinate, which will be completed before sucking board and is a
          operation that
          the robot arm will move to neighbor of tile of last unit to compute the corner position. Then, I implement
          computeCartesianPath in <a href="https://moveit.ros.org/" class="text-blue-600 hover:underline">MoveIt!</a> to
          finish the injecting process as the
          video shows.
        </p>
        <div class="relative w-full mt-2" style="padding-bottom: 56.25%; height: 0;">
          <iframe src="https://www.youtube.com/embed/y2w0eC6z7mc"
            class="absolute top-0 left-0 w-full h-full" frameborder="0"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        <p class="mt-2">
          The final result could be like the picture below with the help of scraper.
        </p>
        <img src="@/assets/images/injecting.jpeg" alt="Result" class="mt-2 max-w-full" />

        <p class="mt-4">
          This is our group photo.
        </p>
        <img src="@/assets/images/roboticplus.jpeg" alt="RoboticPlus" class="mt-2 max-w-full" />
      </section>

      <section class="border-t border-gray-200 pt-6">
        <h3 id="iot_project" class="text-lg font-semibold">IoT Project: Smart Movable Trash Bin</h3>
        <div class="relative w-full mt-2" style="padding-bottom: 56.25%; height: 0;">
          <iframe src="https://www.youtube.com/embed/5ggCkcRDrto"
            class="absolute top-0 left-0 w-full h-full" frameborder="0"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        <p class="mt-2">
          Smart Movable Trash-Bin (SMTB) is an IoT solution to current trash sorting problem in office. Currently, the
          trash sorting is a trend and a remedial action to save the environment, which becomes a policy of great
          significance in numerous countries and cities. However, people, especially the officer in office, who are not
          going to have a single trash-bin suffers from the trash sorting and often feel confused about the trash type
          every time they are going to drop trash. Our project aims to automate the process in the origin, which means
          when people drop trash, the trash is accurately identifed as the trash type, including metal, plastic, glass,
          paper, cardboard (recyclable in most regions) and trash (unrecyclable). In addition, our target is to make it
          as a intelligent project. Since in most cases, officers should share a common trash-bin and need to move when
          they have trash. But with our design, user can directly use phone to call our SMTB to come to his/her postion
          and user can simply put trash in and SMTB intelligently identify the trash type and go back orginal place for
          sorting and collecting.
        </p>
        <p class="mt-2">
          With SMTB, officers will benefit from simplification of dropping trash and trash sorting; office will become
          more clean; government will benefit from the energy conservation resulting from resource reuse. We hope our
          project will make a difference to the world.
        </p>
        <p class="mt-2">
          The details of this project is shown in the page
          <router-link to="/project/iot" class="text-blue-600 hover:underline">IOT SMART MOVABLE TRASH BIN
            PROJECT</router-link>
          and the code in
          <a href="https://github.com/qw2261/IoT-Project" class="text-blue-600 hover:underline">my github
            repository</a>.
        </p>
      </section>

    </div>
  </div>
</template>
