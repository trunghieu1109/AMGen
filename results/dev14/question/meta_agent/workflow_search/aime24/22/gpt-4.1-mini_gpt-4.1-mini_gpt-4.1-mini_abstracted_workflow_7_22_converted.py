async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_0_1 = "Sub-task 0_1: Formally represent the list elements as positive integers and express the sum constraint: the sum of all elements equals 30. Avoid attempting to solve or enumerate lists at this stage; focus solely on formalizing the sum constraint."
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing sum constraint, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_0_2 = "Sub-task 0_2: Formally define the unique mode constraint: identify that the integer 9 appears more times than any other integer in the list, and no other integer has the same frequency as 9. Avoid assuming the frequency count; just represent the uniqueness condition."
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, formalizing unique mode constraint, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_0_3 = "Sub-task 0_3: Formally characterize the median constraint: the median is a positive integer that does not appear in the list. Clarify implications on list length (likely even) and ordering without enumerating values or lengths."
    subtask_desc_0_3 = {
        "subtask_id": "subtask_0_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, formalizing median constraint, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0_3: ", sub_tasks[-1])

    N_sc_0_4 = self.max_sc
    cot_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_4)]
    possible_answers_0_4 = []
    possible_thinkings_0_4 = []
    cot_sc_instruction_0_4 = "Sub-task 0_4: Synthesize the constraints from subtasks 0_1, 0_2, and 0_3 to identify logical implications and restrictions on the list structure, such as possible list lengths, frequency bounds, and median positioning. Avoid enumerating specific lists here."
    subtask_desc_0_4 = {
        "subtask_id": "subtask_0_4",
        "instruction": cot_sc_instruction_0_4,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content, thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_4):
        thinking_i, answer_i = await cot_agents_0_4[i]([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3], cot_sc_instruction_0_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_4[i].id}, synthesizing constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_4.append(answer_i)
        possible_thinkings_0_4.append(thinking_i)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + possible_thinkings_0_4, "Sub-task 0_4: Synthesize and choose the most consistent and correct logical implications for the list structure.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing constraints, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 0_4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0_4: ", sub_tasks[-1])

    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_1_1 = "Sub-task 1_1: Enumerate possible list lengths consistent with the median being a positive integer not in the list, focusing on even lengths and median calculation rules. Avoid generating full lists yet; just identify feasible lengths."
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_4], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerating possible list lengths, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_1_2 = "Sub-task 1_2: Identify possible frequency counts for the mode 9 that satisfy the uniqueness condition, considering the total sum constraint and list length candidates. Avoid full list enumeration; focus on frequency feasibility."
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_4.content, thinking_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_0_4, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, identifying mode 9 frequency counts, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    N_sc_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    cot_sc_instruction_1_3 = "Sub-task 1_3: Enumerate candidate lists of positive integers that sum to 30, have 9 as the unique mode with identified frequency, and have a median that is a positive integer not in the list. Use constraints from previous subtasks to prune the search space. Avoid computing sum of squares at this stage."
    subtask_desc_1_3 = {
        "subtask_id": "subtask_1_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_1.content, thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_3):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_1_1, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, enumerating candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 1_3: Synthesize and choose the most consistent candidate lists.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing candidate lists, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1_3: ", sub_tasks[-1])

    N_sc_1_4 = self.max_sc
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_4)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    cot_sc_instruction_1_4 = "Sub-task 1_4: Verify that each candidate list from subtask_1_3 satisfies all constraints simultaneously, especially the median condition that the median integer is not in the list. Discard invalid candidates."
    subtask_desc_1_4 = {
        "subtask_id": "subtask_1_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_4):
        thinking_i, answer_i = await cot_agents_1_4[i]([taskInfo, thinking_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, verifying candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_4.append(answer_i)
        possible_thinkings_1_4.append(thinking_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_thinkings_1_4, "Sub-task 1_4: Synthesize and choose the valid candidate lists that satisfy all constraints.", is_sub_task=True)
    agents.append(f"Final Decision agent, verifying candidate lists, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 1_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1_4: ", sub_tasks[-1])

    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_2_1 = "Sub-task 2_1: For the validated list(s) from subtask_1_4, compute the sum of the squares of all items in the list. Avoid re-verifying constraints; focus solely on the arithmetic computation."
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_4], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing sum of squares, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = "Sub-task 2_2: Analyze the computed sum of squares to confirm uniqueness or identify if multiple solutions exist. Provide reasoning or reflection on the result's consistency with the problem constraints." + reflect_inst_2_2
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, analyzing sum of squares uniqueness, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining analysis, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2_2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
