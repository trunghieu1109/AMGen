async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage1 = "Sub-task 1: Determine whether the given stability constants β1=9, β2=40, β3=63, β4=16 are stepwise or cumulative constants. Justify the assumption and convert them into cumulative stability constants κ1, κ2, κ3, κ4 by sequential multiplication. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1 = self.max_round
    all_thinking_stage1 = [[] for _ in range(N_max_stage1)]
    all_answer_stage1 = [[] for _ in range(N_max_stage1)]
    subtask_desc1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": debate_instr_stage1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1):
        for i, agent in enumerate(debate_agents_stage1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage1[r-1] + all_answer_stage1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1[r].append(thinking)
            all_answer_stage1[r].append(answer)
    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1([taskInfo] + all_thinking_stage1[-1] + all_answer_stage1[-1], "Sub-task 1: Synthesize and choose the most consistent and correct solution for stability constants." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_stage1_2 = "Sub-task 2: Summarize and classify all given chemical data including total cobalt concentration (10^-2 M), initial thiocyanate concentration (0.1 M), and the cumulative stability constants derived from Sub-task 1. Identify all relevant species and state assumptions such as ideal solution behavior."
    N_sc = self.max_sc
    cot_agents_stage1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1_2 = []
    possible_thinkings_stage1_2 = []
    subtask_desc2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_stage1_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_stage1_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_stage1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_stage1_2.append(answer2)
        possible_thinkings_stage1_2.append(thinking2)
    final_decision_agent_stage1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_stage1_2([taskInfo, thinking1, answer1] + possible_thinkings_stage1_2 + possible_answers_stage1_2, "Sub-task 2: Synthesize and choose the most consistent and correct summary of chemical data." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage2_1 = "Sub-task 1: Formulate the full system of equilibrium expressions and mass-balance equations for cobalt and thiocyanate species using the cumulative stability constants. Include expressions for concentrations of each complex species as functions of free Co2+ and free SCN-. Incorporate mass-balance constraints for total cobalt and thiocyanate, accounting for ligand consumption. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage2_1 = self.max_round
    all_thinking_stage2_1 = [[] for _ in range(N_max_stage2_1)]
    all_answer_stage2_1 = [[] for _ in range(N_max_stage2_1)]
    subtask_desc3 = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate_instr_stage2_1,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage2_1):
        for i, agent in enumerate(debate_agents_stage2_1):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_stage2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking_stage2_1[r-1] + all_answer_stage2_1[r-1]
                thinking3, answer3 = await agent(input_infos, debate_instr_stage2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage2_1[r].append(thinking3)
            all_answer_stage2_1[r].append(answer3)
    final_decision_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_stage2_1([taskInfo, thinking2, answer2] + all_thinking_stage2_1[-1] + all_answer_stage2_1[-1], "Sub-task 1: Synthesize and choose the most consistent and correct equilibrium formulation." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_stage2_2 = "Sub-task 2: Numerically solve the nonlinear system of equilibrium and mass-balance equations formulated in Sub-task 1 to determine free Co2+, free SCN-, and all Co(SCN)n complex concentrations. Use appropriate numerical methods to ensure accuracy and convergence."
    N_sc_stage2_2 = self.max_sc
    cot_agents_stage2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_stage2_2)]
    possible_answers_stage2_2 = []
    possible_thinkings_stage2_2 = []
    subtask_desc4 = {
        "subtask_id": "stage2_subtask2",
        "instruction": cot_sc_instruction_stage2_2,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_stage2_2):
        thinking4, answer4 = await cot_agents_stage2_2[i]([taskInfo, thinking3, answer3], cot_sc_instruction_stage2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2_2[i].id}, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_stage2_2.append(answer4)
        possible_thinkings_stage2_2.append(thinking4)
    final_decision_agent_stage2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_stage2_2([taskInfo, thinking3, answer3] + possible_thinkings_stage2_2 + possible_answers_stage2_2, "Sub-task 2: Synthesize and choose the most consistent and correct numerical solution." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst_stage3_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage3_1 = "Sub-task 1: Calculate the percentage of the blue dithiocyanato cobalt(II) complex (Co(SCN)2) relative to the total cobalt concentration using the concentrations obtained from the numerical solution. Verify the result by cross-checking with the total cobalt mass balance and ensure consistency." + reflect_inst_stage3_1
    cot_agent_stage3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage3_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage3_1 = self.max_round
    cot_inputs_stage3_1 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "stage3_subtask1",
        "instruction": cot_reflect_instruction_stage3_1,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_stage3_1(cot_inputs_stage3_1, cot_reflect_instruction_stage3_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage3_1.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_stage3_1):
        feedback, correct = await critic_agent_stage3_1([taskInfo, thinking5, answer5], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage3_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage3_1.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_stage3_1(cot_inputs_stage3_1, cot_reflect_instruction_stage3_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage3_1.id}, refining thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflect_inst_stage3_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage3_2 = "Sub-task 2: Compare the calculated percentage of the dithiocyanato complex with the provided answer choices (16.9%, 42.3%, 25.6%, 38.1%) and select the correct option based on the numerical result. Document the reasoning and verification steps to justify the final choice." + reflect_inst_stage3_2
    cot_agent_stage3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage3_2 = self.max_round
    cot_inputs_stage3_2 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "stage3_subtask2",
        "instruction": cot_reflect_instruction_stage3_2,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_stage3_2(cot_inputs_stage3_2, cot_reflect_instruction_stage3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage3_2.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_stage3_2):
        feedback, correct = await critic_agent_stage3_2([taskInfo, thinking6, answer6], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage3_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage3_2.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_stage3_2(cot_inputs_stage3_2, cot_reflect_instruction_stage3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage3_2.id}, refining thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
