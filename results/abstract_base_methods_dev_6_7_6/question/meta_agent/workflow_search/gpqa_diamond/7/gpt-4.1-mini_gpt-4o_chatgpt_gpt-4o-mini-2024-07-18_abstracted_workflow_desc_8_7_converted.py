async def forward_7(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = (
        "Sub-task 1: Analyze the resistance levels of single-gene knock-out mutants (g1, g2, g3) relative to the wild-type control to identify which genes significantly reduce resistance and may be critical for anthracnose resistance. "
        "Use the provided resistance data and reason step-by-step about each gene's impact."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing single-gene knock-out mutants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the resistance levels of double-gene knock-out mutants (g1g2, g1g3, g2g3) relative to the wild-type control and single mutants to detect interaction effects between genes on anthracnose resistance. "
        "Consider multiple reasoning paths to ensure consistency."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing double-gene knock-out mutants, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    cot_reflect_instruction_3 = (
        "Sub-task 3: Determine which gene is likely a transcription factor acting upstream by interpreting resistance data, focusing on genes whose knockout causes complete loss of resistance or epistatic effects. "
        "Use clear definitions of transcription factor roles and genetic interaction types, and refine reasoning based on previous subtasks."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining transcription factor gene, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the transcription factor identification and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining transcription factor identification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    cot_sc_instruction_4 = (
        "Sub-task 4: Identify and classify the genetic interaction between G1 and G3 (e.g., redundancy, epistasis, pleiotropy, or additive effects) by applying rigorous definitions and analyzing resistance patterns in single and double mutants. "
        "Use a self-consistency chain-of-thought approach to evaluate multiple reasoning paths and select the most consistent interpretation."
    )
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, classifying genetic interaction between G1 and G3, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    cot_instruction_5a = (
        "Sub-task 5a: Map the key experimental findings from subtasks 1 to 4 onto a truth table comparing the four multiple-choice options, focusing on transcription factor identity and gene interaction types. "
        "Provide a clear and structured mapping to facilitate decision-making."
    )
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, mapping key findings to truth table, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    debate_instruction_5b = (
        "Sub-task 5b: Select the multiple-choice answer that best matches the experimental data and genetic interaction analysis using weighted aggregation. "
        "Prioritize correct identification of the transcription factor and accurate interaction classification. "
        "If votes conflict, select the option aligning with all critical criteria."
    )
    debate_agents_5b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5b = self.max_round
    all_thinking5b = [[] for _ in range(N_max_5b)]
    all_answer5b = [[] for _ in range(N_max_5b)]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": debate_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5b):
        for i, agent in enumerate(debate_agents_5b):
            if r == 0:
                thinking5b, answer5b = await agent([taskInfo, thinking5a, answer5a], debate_instruction_5b, r, is_sub_task=True)
            else:
                input_infos_5b = [taskInfo, thinking5a, answer5a] + all_thinking5b[r-1] + all_answer5b[r-1]
                thinking5b, answer5b = await agent(input_infos_5b, debate_instruction_5b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final answer selection, thinking: {thinking5b.content}; answer: {answer5b.content}")
            all_thinking5b[r].append(thinking5b)
            all_answer5b[r].append(answer5b)
    final_decision_agent_5b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5b, answer5b = await final_decision_agent_5b([taskInfo] + all_thinking5b[-1] + all_answer5b[-1], "Sub-task 5b: Make final decision on the multiple-choice answer that best fits the experimental data and genetic interaction analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final conclusion, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    cot_reflect_instruction_6 = (
        "Sub-task 6: Perform a final reflection and validation step to cross-check the selected answer against the original correct index and ensure logical consistency and alignment with experimental data and definitions. "
        "Provide justification for the final choice and confirm its correctness."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5b, answer5b]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, final reflection and validation, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the final answer validation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final answer validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 1: ", sub_tasks[0])
    print("Step 2: ", sub_tasks[1])
    print("Step 3: ", sub_tasks[2])
    print("Step 4: ", sub_tasks[3])
    print("Step 5a: ", sub_tasks[4])
    print("Step 5b: ", sub_tasks[5])
    print("Step 6: ", sub_tasks[6])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
