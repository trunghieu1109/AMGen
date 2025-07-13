async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 1: Extract and define the stereochemical features of the starting materials and the reaction conditions, including the stereospecificity of mCPBA epoxidation on (E)- and (Z)-oct-4-ene and the nature of the products after aqueous acid treatment. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking_0, answer_0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking_0, answer_0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_0.content}; answer: {answer_0.content}")
            all_thinking_0[r].append(thinking_0)
            all_answer_0[r].append(answer_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0, answer_0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], "Sub-task 1: Extract and define stereochemical features. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1 = "Sub-task 2: Determine the number and stereochemical nature (diastereomers and enantiomers) of the epoxide products formed from each alkene isomer after epoxidation and acid treatment, based on the output from Sub-task 1."
    N_sc_1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo, thinking_0, answer_0] + possible_thinkings_1 + possible_answers_1, "Sub-task 2: Synthesize and choose the most consistent answer for the stereochemical nature and number of epoxide products.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 3: Combine the stereoisomeric product sets from both (E)- and (Z)-oct-4-ene reactions to enumerate the total distinct stereoisomers present in the combined mixture." + reflect_inst_2
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1]
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, thinking: {thinking_2.content}; answer: {answer_2.content}")
    for i in range(N_max_2):
        feedback_2, correct_2 = await critic_agent_2([taskInfo, thinking_2, answer_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback_2.content}; correct: {correct_2.content}")
        if correct_2.content == "True":
            break
        cot_inputs_2.extend([thinking_2, answer_2, feedback_2])
        thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3a = "Sub-task 4.1: Predict the chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column, focusing on the number of peaks expected based on diastereomer separation and enantiomer co-elution. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking_3a = [[] for _ in range(N_max_3a)]
    all_answer_3a = [[] for _ in range(N_max_3a)]
    subtask_desc_3a = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_3a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking_3a, answer_3a = await agent([taskInfo, thinking_2, answer_2], debate_instr_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking_2, answer_2] + all_thinking_3a[r-1] + all_answer_3a[r-1]
                thinking_3a, answer_3a = await agent(input_infos_3a, debate_instr_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
            all_thinking_3a[r].append(thinking_3a)
            all_answer_3a[r].append(answer_3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3a, answer_3a = await final_decision_agent_3a([taskInfo, thinking_2, answer_2] + all_thinking_3a[-1] + all_answer_3a[-1], "Sub-task 4.1: Predict chromatographic behavior on achiral HPLC. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instr_3b = "Sub-task 4.2: Predict the chromatographic behavior of the combined product mixture on a chiral HPLC column, focusing on the number of peaks expected based on separation of both diastereomers and enantiomers. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking_3b = [[] for _ in range(N_max_3b)]
    all_answer_3b = [[] for _ in range(N_max_3b)]
    subtask_desc_3b = {
        "subtask_id": "subtask_2",
        "instruction": debate_instr_3b,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking_3b, answer_3b = await agent([taskInfo, thinking_2, answer_2], debate_instr_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking_2, answer_2] + all_thinking_3b[r-1] + all_answer_3b[r-1]
                thinking_3b, answer_3b = await agent(input_infos_3b, debate_instr_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
            all_thinking_3b[r].append(thinking_3b)
            all_answer_3b[r].append(answer_3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3b, answer_3b = await final_decision_agent_3b([taskInfo, thinking_2, answer_2] + all_thinking_3b[-1] + all_answer_3b[-1], "Sub-task 4.2: Predict chromatographic behavior on chiral HPLC. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3b, answer_3b, sub_tasks, agents)
    return final_answer, logs
