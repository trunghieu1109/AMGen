async def forward_82(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and combine initial quantitative inputs and identify key parameters

    # Sub-task 1: Calculate new concentration of acetic acid after dilution
    cot_instruction_1 = (
        "Sub-task 1: Calculate the new concentration of acetic acid after dilution by combining the initial volume and molarity of acetic acid with the volume of water added. "
        "Consider total volume change and provide the diluted concentration."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculated diluted concentration, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and classify key chemical parameters and constants
    cot_instruction_2 = (
        "Sub-task 2: Identify and classify key chemical parameters relevant to the problem including Ka of acetic acid, temperature (25 °C), and titration points (25% titration and equivalence point). "
        "Summarize these constants for use in further calculations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, identified key chemical parameters, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Calculate moles and concentrations at titration points

    # Sub-task 3: Calculate initial moles of acetic acid before and after dilution
    cot_sc_instruction_3 = (
        "Sub-task 3: Calculate the initial moles of acetic acid before dilution and after dilution using the diluted concentration and total volume from Sub-task 1. "
        "Perform self-consistency checks to ensure mole calculations are accurate."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating moles of acetic acid, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer
    most_common_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_3]
    answer3 = answermapping_3[most_common_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine moles of NaOH added at 25% titration and equivalence point
    cot_sc_instruction_4 = (
        "Sub-task 4: Using the initial moles of acetic acid from Sub-task 3, calculate the moles of NaOH added at 25% titration and at equivalence point. "
        "Use self-consistency to verify mole calculations."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, calculating moles of NaOH added, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_4]
    answer4 = answermapping_4[most_common_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate concentrations at 25% titration point
    cot_instruction_5 = (
        "Sub-task 5: Calculate the concentrations of acetic acid, acetate ion, and remaining species at 25% titration point using moles from Sub-tasks 3 and 4 and total volume after titration. "
        "Use Chain-of-Thought reasoning to detail the calculation steps."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating concentrations at 25% titration, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Calculate concentrations at equivalence point
    cot_instruction_6 = (
        "Sub-task 6: Calculate the concentrations of species at the equivalence point considering complete neutralization and formation of acetate ion, using moles from Sub-tasks 3 and 4. "
        "Use Chain-of-Thought reasoning to explain the process."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating concentrations at equivalence point, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Calculate pH values and compare with choices

    # Sub-task 7: Calculate pH at 25% titration using Henderson-Hasselbalch equation
    cot_instruction_7 = (
        "Sub-task 7: Calculate the pH at 25% titration point using the Henderson-Hasselbalch equation with concentrations of acetic acid and acetate ion from Sub-task 5 and Ka from Sub-task 2. "
        "Explain the calculation steps clearly."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5, thinking2, answer2], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculating pH at 25% titration, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Calculate pH at equivalence point by hydrolysis of acetate ion
    cot_instruction_8 = (
        "Sub-task 8: Calculate the pH at the equivalence point by determining the hydrolysis of acetate ion using Kb derived from Ka (from Sub-task 2) and concentrations from Sub-task 6. "
        "Provide detailed reasoning steps."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6, thinking2, answer2], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, calculating pH at equivalence point, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Compare calculated pH values with given multiple-choice options
    debate_instruction_9 = (
        "Sub-task 9: Based on the calculated pH values at 25% titration and equivalence point from Sub-tasks 7 and 8, compare these values with the given multiple-choice options. "
        "Use debate agents to argue which choice best matches the calculated pH values and select the correct answer."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            input_infos_9 = [taskInfo, thinking7, answer7, thinking8, answer8]
            if r > 0:
                input_infos_9 += all_thinking9[r-1] + all_answer9[r-1]
            thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing pH values with choices, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct multiple-choice answer based on pH calculations.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting correct answer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer
