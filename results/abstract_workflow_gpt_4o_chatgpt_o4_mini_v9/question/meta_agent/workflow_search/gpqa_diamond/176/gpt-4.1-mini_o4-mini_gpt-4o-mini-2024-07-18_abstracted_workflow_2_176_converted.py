async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze and classify the given physical parameters of the two stars
    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the given physical parameters of the two stars: radius ratio (Star_1 radius is 1.5 times Star_2), "
        "mass ratio (Star_1 mass is 1.5 times Star_2), and the fact that their peak wavelengths are the same (implying equal surface temperatures). "
        "Identify how these parameters relate to luminosity calculation."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Interpret significance of same peak wavelength (equal surface temperatures) using Self-Consistency CoT
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, interpret the significance of the stars having the same peak wavelength in their spectra, "
        "and deduce that this implies equal surface temperatures based on Wien's displacement law."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, interpreting equal peak wavelength, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer (self-consistency)
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_content = thinkingmapping_2[answer2_content].content
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_content}; answer - {answer2_content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Transform, Extract Features, Evaluate, and Select Elements
    # Sub-task 3: Calculate luminosity ratio using radius ratio and equal temperature condition
    cot_instruction_3 = (
        "Sub-task 3: Using the radius ratio and the equal temperature condition from previous subtasks, calculate the luminosity ratio of Star_1 to Star_2 assuming black body radiation, "
        "applying the Stefan-Boltzmann law (L = 4πR^2σT^4)."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answermapping_2[answer2_content]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating luminosity ratio, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Evaluate effect of radial velocities on observed luminosities
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the effect of the given radial velocities (0 km/s for Star_1 and 700 km/s for Star_2) on the observed luminosities, "
        "and determine if any Doppler or relativistic corrections are necessary for the luminosity comparison."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answermapping_2[answer2_content]], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating velocity effects, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Combine results from luminosity ratio and velocity effect to finalize factor and compare to choices using Debate
    debate_roles = ["Pro-Luminosity Ratio", "Pro-Velocity Effect"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking3, answer3, thinking4, answer4],
                    "Sub-task 5: Combine luminosity ratio and velocity effect evaluation to finalize the factor by which Star_1's luminosity is greater than Star_2's, and compare this factor to the provided multiple-choice options.",
                    r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, 
                                                "Sub-task 5: Refine final luminosity factor and choice comparison based on debate feedback.",
                                                r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final luminosity factor, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on the factor by which Star_1's luminosity is greater than Star_2's and select the closest multiple-choice option.",
        is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final luminosity factor and choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
