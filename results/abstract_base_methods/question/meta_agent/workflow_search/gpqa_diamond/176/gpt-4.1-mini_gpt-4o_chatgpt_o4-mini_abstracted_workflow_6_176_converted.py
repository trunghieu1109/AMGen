async def forward_176(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    c = 3e5
    stage1_subtask1_instruction = (
        "Sub-task 1: Collect and explicitly state all relevant observational data for both stars, "
        "including observed peak wavelengths and radial velocities, to establish the initial physical context for analysis."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": stage1_subtask1_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], stage1_subtask1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, collecting observational data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    stage1_subtask2_instruction = (
        "Sub-task 2: Correct the observed peak wavelengths of both stars for Doppler shifts using their radial velocities "
        "and the formula lambda_rest = lambda_obs / (1 + v_radial/c), to determine the intrinsic (rest-frame) peak wavelengths."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": stage1_subtask2_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], stage1_subtask2_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, correcting wavelengths for Doppler shift, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    stage1_subtask3_instruction = (
        "Sub-task 3: Calculate the intrinsic temperatures of both stars by applying Wien's displacement law "
        "to the Doppler-corrected peak wavelengths obtained in Sub-task 2."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": stage1_subtask3_instruction,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], stage1_subtask3_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating intrinsic temperatures, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    stage1_subtask4_instruction = (
        "Sub-task 4: Critically evaluate and reflect on the temperature results to verify that Doppler corrections have been properly applied "
        "and that the temperatures are physically consistent with the given data, ensuring no relevant physical effects are overlooked."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": stage1_subtask4_instruction,
        "context": ["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2", "thinking and answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, stage1_subtask4_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, reflecting on temperature results, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4], "Review the temperature calculation and Doppler correction for physical accuracy and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback on temperature reflection, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, stage1_subtask4_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining temperature reflection, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    stage2_subtask5_instruction = (
        "Sub-task 5: Using the intrinsic temperatures from Sub-task 3 and the given radius ratio (Star_1 radius is 1.5 times Star_2 radius), "
        "calculate the luminosity ratio of Star_1 to Star_2 applying the Stefan-Boltzmann law (L = 4πR²σT⁴)."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": stage2_subtask5_instruction,
        "context": ["user query", "thinking and answer of subtask 3", "thinking and answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    N5 = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N5):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking3, answer3, thinking4, answer4], stage2_subtask5_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, calculating luminosity ratio, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    answer5_content = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[answer5_content]
    answer5 = answermapping_5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    stage2_subtask6_instruction = (
        "Sub-task 6: Perform a final validation by cross-checking the entire reasoning chain, including Doppler corrections, temperature calculations, "
        "and luminosity ratio computations, to ensure physical accuracy and completeness before selecting the answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": stage2_subtask6_instruction,
        "context": ["user query", "all previous thinking and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, stage2_subtask6_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, final validation of reasoning chain, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6], "Review the entire reasoning chain for physical accuracy and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback on final validation, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, stage2_subtask6_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    stage3_subtask7_instruction = (
        "Sub-task 7: Compare the validated luminosity ratio with the provided multiple-choice options and select the closest matching factor as the final answer."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": stage3_subtask7_instruction,
        "context": ["user query", "thinking and answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], stage3_subtask7_instruction, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, stage3_subtask7_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting closest luminosity ratio factor, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the closest matching luminosity ratio factor.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting luminosity ratio factor, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs