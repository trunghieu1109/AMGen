async def forward_123(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and Classify Key Physical Parameters
    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the problem context where particles are produced at the center of a spherical detector "
        "with radius 30 meters, traveling at ultra-relativistic velocities with Lorentz factor ~20, and on average one third "
        "reach the detector walls before decaying. Identify and classify key physical parameters such as particle decay behavior, "
        "Lorentz factor, detector geometry, and fraction of particles reaching the walls."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing problem context, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Based on the analysis from Sub-task 1, classify the relationship between the Lorentz factor and the fraction "
        "of particles reaching the detector walls, considering relativistic time dilation effects on particle decay length and "
        "how it affects survival probability over the 30 meter radius distance."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    thinking_mapping_0_2 = {}
    answer_mapping_0_2 = {}
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, classifying Lorentz factor and survival fraction, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i.content)
        thinking_mapping_0_2[answer_i.content] = thinking_i
        answer_mapping_0_2[answer_i.content] = answer_i
    # Choose the most common answer
    most_common_answer_0_2 = Counter(possible_answers_0_2).most_common(1)[0][0]
    thinking_0_2 = thinking_mapping_0_2[most_common_answer_0_2]
    answer_0_2 = answer_mapping_0_2[most_common_answer_0_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Assess Effect of Changing Lorentz Factor on Survival Probability
    cot_reflect_instruction_1 = (
        "Sub-task 3: Assess how increasing the Lorentz factor from ~20 to a higher value affects the survival probability "
        "(fraction) of particles reaching the detector walls, given the fixed detector radius of 30 meters and initial condition "
        "that one third of particles reach the walls at Lorentz factor ~20."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking_0_2, answer_0_2]
    thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, cot_reflect_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, assessing Lorentz factor impact, thinking: {thinking_1.content}; answer: {answer_1.content}")
    for i in range(N_max_1):
        feedback_1, correct_1 = await critic_agent_1([taskInfo, thinking_1, answer_1],
                                                    "Critically evaluate the assessment of Lorentz factor impact on survival probability, mathematical correctness, and completeness.",
                                                    i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, providing feedback, thinking: {feedback_1.content}; answer: {correct_1.content}")
        if correct_1.content == "True":
            break
        cot_inputs_1.extend([thinking_1, answer_1, feedback_1])
        thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, cot_reflect_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining assessment, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Derive Mathematical Expression and Calculate Required Lorentz Factor
    debate_instruction_2_4 = (
        "Sub-task 4: Derive the mathematical expression relating the survival fraction of particles reaching the detector walls "
        "to the Lorentz factor, using the exponential decay law modified by relativistic time dilation and the known detector radius."
    )
    debate_agents_2_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_4 = self.max_round
    all_thinking_2_4 = [[] for _ in range(N_max_2_4)]
    all_answer_2_4 = [[] for _ in range(N_max_2_4)]
    for r in range(N_max_2_4):
        for i, agent in enumerate(debate_agents_2_4):
            input_infos_2_4 = [taskInfo, thinking_1, answer_1]
            if r > 0:
                input_infos_2_4 += all_thinking_2_4[r-1] + all_answer_2_4[r-1]
            thinking_2_4, answer_2_4 = await agent(input_infos_2_4, debate_instruction_2_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving expression, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
            all_thinking_2_4[r].append(thinking_2_4)
            all_answer_2_4[r].append(answer_2_4)
    final_decision_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_4, answer_2_4 = await final_decision_agent_2_4([taskInfo] + all_thinking_2_4[-1] + all_answer_2_4[-1],
                                                              "Sub-task 4: Make final decision on the derived mathematical expression.",
                                                              is_sub_task=True)
    agents.append(f"Final Decision agent on expression derivation, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    print("Step 2.4: ", sub_tasks[-1])

    cot_reflect_instruction_2_5 = (
        "Sub-task 5: Using the derived expression and the initial condition (one third survival at Lorentz factor ~20), "
        "calculate the Lorentz factor required to achieve about two thirds survival fraction of particles reaching the detector walls."
    )
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_5 = self.max_round
    cot_inputs_2_5 = [taskInfo, thinking_2_4, answer_2_4]
    thinking_2_5, answer_2_5 = await cot_agent_2_5(cot_inputs_2_5, cot_reflect_instruction_2_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_5.id}, calculating required Lorentz factor, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    for i in range(N_max_2_5):
        feedback_2_5, correct_2_5 = await critic_agent_2_5([taskInfo, thinking_2_5, answer_2_5],
                                                          "Critically evaluate the calculation of required Lorentz factor, mathematical correctness, and completeness.",
                                                          i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_5.id}, providing feedback, thinking: {feedback_2_5.content}; answer: {correct_2_5.content}")
        if correct_2_5.content == "True":
            break
        cot_inputs_2_5.extend([thinking_2_5, answer_2_5, feedback_2_5])
        thinking_2_5, answer_2_5 = await cot_agent_2_5(cot_inputs_2_5, cot_reflect_instruction_2_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_5.id}, refining calculation, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Compare Calculated Lorentz Factor with Multiple Choice Options
    debate_instruction_3_6 = (
        "Sub-task 6: Compare the calculated Lorentz factor from Sub-task 5 with the provided multiple-choice options (28, 40, 68, 54) "
        "and identify the closest matching choice."
    )
    debate_agents_3_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_6 = self.max_round
    all_thinking_3_6 = [[] for _ in range(N_max_3_6)]
    all_answer_3_6 = [[] for _ in range(N_max_3_6)]
    for r in range(N_max_3_6):
        for i, agent in enumerate(debate_agents_3_6):
            input_infos_3_6 = [taskInfo, thinking_2_5, answer_2_5]
            if r > 0:
                input_infos_3_6 += all_thinking_3_6[r-1] + all_answer_3_6[r-1]
            thinking_3_6, answer_3_6 = await agent(input_infos_3_6, debate_instruction_3_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing Lorentz factor, thinking: {thinking_3_6.content}; answer: {answer_3_6.content}")
            all_thinking_3_6[r].append(thinking_3_6)
            all_answer_3_6[r].append(answer_3_6)
    final_decision_agent_3_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_6, answer_3_6 = await final_decision_agent_3_6([taskInfo] + all_thinking_3_6[-1] + all_answer_3_6[-1],
                                                              "Sub-task 6: Make final decision on the closest matching Lorentz factor choice.",
                                                              is_sub_task=True)
    agents.append(f"Final Decision agent on Lorentz factor choice, thinking: {thinking_3_6.content}; answer: {answer_3_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_3_6.content}; answer - {answer_3_6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_6, answer_3_6, sub_tasks, agents)
    return final_answer
