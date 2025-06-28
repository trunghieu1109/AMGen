async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze the key chemical concepts involved
    cot_instruction_1 = (
        "Sub-task 1: Analyze the query to identify and classify the key chemical concepts involved: "
        "thermodynamic oxidizing strength of oxygen in basic solutions, and kinetic reaction rate of oxygen in acidic solutions. "
        "Understand the meaning of thermodynamically stronger/weaker oxidant and kinetically faster/slower reaction."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing key chemical concepts, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract and clarify definitions and criteria for thermodynamic oxidizing strength and kinetic reaction rates
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, extract and clarify the definitions and criteria for thermodynamic oxidizing strength "
        "and kinetic reaction rates in electrochemistry, specifically for oxygen in basic vs acidic solutions. "
        "Review standard electrochemical principles and redox potentials relevant to oxygen."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, extracting definitions and criteria, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for consistency
    counter_2 = Counter(possible_answers_2)
    answer2_content = counter_2.most_common(1)[0][0]
    thinking2_content = thinkingmapping_2[answer2_content].content
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_content}; answer - {answer2_content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Evaluate thermodynamic and kinetic properties
    # Sub-task 3: Evaluate thermodynamic oxidizing strength of oxygen in basic solutions
    cot_instruction_3 = (
        "Sub-task 3: Evaluate the thermodynamic oxidizing strength of oxygen in basic solutions by analyzing standard electrode potentials "
        "and related thermodynamic data to determine if oxygen acts as a stronger or weaker oxidant under these conditions."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answermapping_2[answer2_content]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, evaluating thermodynamic oxidizing strength, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate kinetic reaction rate of oxygen in acidic solutions
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the kinetic reaction rate of oxygen in acidic solutions by analyzing reaction mechanisms, activation energy, "
        "and typical reaction rates to determine if oxygen reacts faster or slower in acidic media."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answermapping_2[answer2_content]], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating kinetic reaction rate, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Integrate findings and select correct answer
    # Sub-task 5: Integrate thermodynamic and kinetic evaluations
    debate_instruction_5 = (
        "Sub-task 5: Integrate the findings from thermodynamic oxidizing strength and kinetic reaction rate evaluations to identify the correct "
        "combination of descriptors (weaker/stronger and faster/slower) that completes the test statement about oxygen's behavior in basic and acidic solutions."
    )
    debate_roles = ["Thermodynamics Expert", "Kinetics Expert"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating thermodynamic and kinetic findings, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct combination of descriptors.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct combination, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Match integrated conclusion with provided multiple-choice options
    cot_instruction_6 = (
        "Sub-task 6: Match the integrated conclusion from Sub-task 5 with the provided multiple-choice options to select the correct answer choice "
        "that corresponds to the evaluated thermodynamic and kinetic properties of oxygen."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, matching conclusion with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
