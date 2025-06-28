async def forward_29(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Convert emission energy to wavelength and identify emitted light color
    # Sub-task 1: Convert emission energy (2.3393 eV) to wavelength in nm using E = hc/\u03BB
    cot_instruction_1 = (
        "Sub-task 1: Convert the given emission energy of 2.3393 eV into the corresponding wavelength of emitted light in nanometers, "
        "using the relationship E = hc/\u03BB where h is Planck's constant and c is speed of light."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, converting emission energy to wavelength, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the color of emitted light based on wavelength from sub-task 1 using visible spectrum ranges
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the calculated emission wavelength from Sub-task 1, identify the color of the emitted light "
        "using the visible light spectrum ranges (approximate ranges: Violet 380-450 nm, Blue 450-495 nm, Green 495-570 nm, Yellow 570-590 nm, Orange 590-620 nm, Red 620-750 nm)."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying emitted light color, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Majority vote for emitted color
    emitted_color = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[emitted_color]
    answer2_final = answermapping_2[emitted_color]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Determine absorbed light wavelength and color (complementary color of emitted light)
    # Sub-task 3: Determine wavelength and color of absorbed light knowing it is complementary to emitted light
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on the emitted light color identified in Sub-task 2, determine the wavelength and color of light absorbed by the organic compound, "
        "knowing that absorbed light corresponds to the complementary color of the emitted light. Use standard complementary color pairs in visible spectrum."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2_final, answer2_final]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining absorbed light color and wavelength, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Match absorbed light color with given multiple-choice options
    # Sub-task 4: Match absorbed light color with choices (Blue, Yellow, Red, Violet)
    debate_instruction_4 = (
        "Sub-task 4: Based on the absorbed light color determined in Sub-task 3, match it with the given multiple-choice options: Blue, Yellow, Red, Violet, "
        "and select the correct answer."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching absorbed color to choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the absorbed light color choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on absorbed color choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
