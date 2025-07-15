async def forward_195(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Extract and summarize all given physical parameters, assumptions, and the problem context, including the nature of the oscillator, the force law, and relativistic considerations."
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, extracting and summarizing problem context, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    counter_1 = Counter([a.content for a in possible_answers_1])
    most_common_answer_1 = counter_1.most_common(1)[0][0]
    idx_1 = [a.content for a in possible_answers_1].index(most_common_answer_1)
    thinking1_final = possible_thinkings_1[idx_1]
    answer1_final = possible_answers_1[idx_1]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1_final,
        "answer": answer1_final
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Analyze the relationships between the physical quantities (m, k, A, c) and the implications of relativistic dynamics on the harmonic oscillator, focusing on energy relations and velocity constraints."
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1_final, answer1_final],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1_final, answer1_final], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing physical relationships and relativistic implications, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    counter_2 = Counter([a.content for a in possible_answers_2])
    most_common_answer_2 = counter_2.most_common(1)[0][0]
    idx_2 = [a.content for a in possible_answers_2].index(most_common_answer_2)
    thinking2_final = possible_thinkings_2[idx_2]
    answer2_final = possible_answers_2[idx_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2_final,
        "answer": answer2_final
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3 = "Sub-task 3: Classify the problem within the relevant physics subfields and identify key concepts and potential challenges in applying relativistic corrections to the harmonic oscillator." + reflect_inst_3
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking1_final, answer1_final, thinking2_final, answer2_final],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, classifying problem and identifying challenges, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        critic_inst_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], critic_inst_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining classification and challenges, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = "Sub-task 4: Compute or derive the relativistic expression for the maximum speed v_max by applying relativistic energy conservation principles to the harmonic oscillator, incorporating the potential energy and rest energy terms."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking2_final, answer2_final, thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2_final, answer2_final, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, deriving relativistic v_max expression, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    counter_4 = Counter([a.content for a in possible_answers_4])
    most_common_answer_4 = counter_4.most_common(1)[0][0]
    idx_4 = [a.content for a in possible_answers_4].index(most_common_answer_4)
    thinking4_final = possible_thinkings_4[idx_4]
    answer4_final = possible_answers_4[idx_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4_final,
        "answer": answer4_final
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_5 = "Sub-task 5: Compare the derived expression for v_max with the given candidate formulas, verify physical consistency (e.g., v_max < c), and select the correct formula for the maximum speed of the relativistic harmonic oscillator. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instr_5,
        "context": ["user query", thinking4_final, answer4_final],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4_final, answer4_final], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4_final, answer4_final] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidate formulas and physical consistency, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_instr_5 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5_final, answer5_final = await final_decision_agent_5([taskInfo, thinking4_final, answer4_final] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Select correct v_max formula." + final_decision_instr_5, is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct formula, thinking: {thinking5_final.content}; answer: {answer5_final.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_final.content}; answer - {answer5_final.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5_final,
        "answer": answer5_final
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5_final, answer5_final, sub_tasks, agents)
    return final_answer, logs
