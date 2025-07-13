async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Formally represent the digits in the 2x3 grid as variables a,b,c,d,e,f, "
        "where the first row digits are a,b,c and the second row digits are d,e,f. "
        "Define the two 3-digit numbers formed by rows as N1 = 100a + 10b + c and N2 = 100d + 10e + f. "
        "Define the three 2-digit numbers formed by columns as M1 = 10a + d, M2 = 10b + e, M3 = 10c + f. "
        "Validate the sum constraints: N1 + N2 = 999 and M1 + M2 + M3 = 99. "
        "Clarify that digits a,b,c,d,e,f are in [0,9] and leading zeros are allowed as per the example. "
        "Ensure the model is precise and unambiguous for further stages.")

    cot_sc_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0, answer0 = await cot_sc_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, formal representation and validation, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)

    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, 
                                                    "Sub-task 1: Synthesize and choose the most consistent and correct formal representation and validation of the problem.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    reflexion_instruction_1 = (
        "Sub-task 1: Derive composite algebraic equations relating the digits a,b,c,d,e,f from the sum constraints. "
        "Combine constraints to obtain key linear relations such as c + f = 9, d = 1 + b + c with b + c ≤ 8, and 10a + 11b + 10c + e = 89. "
        "Emphasize correctness and completeness to prevent errors in enumeration.")

    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    max_round_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking0, answer0]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": reflexion_instruction_1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "agent_collaboration": "Reflexion"
    }

    thinking1, answer1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, deriving algebraic relations, thinking: {thinking1.content}; answer: {answer1.content}")

    critic_inst_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(max_round_1):
        feedback, correct = await critic_agent_1([taskInfo, thinking1, answer1], critic_inst_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_1.extend([thinking1, answer1, feedback])
        thinking1, answer1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining algebraic relations, thinking: {thinking1.content}; answer: {answer1.content}")

    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 1: Analyze the composite equations derived in stage 1 to infer digit-level constraints and valid ranges for each digit a,b,c,d,e,f. "
        "Identify dependencies among digits and clarify feasible domains, highlighting key parameters a,b,c,e and their interrelations. "
        "Prepare for systematic enumeration in the next stage.")

    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking0, answer0, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, analyzing digit constraints, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking0, answer0, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, 
                                                    "Sub-task 3: Synthesize and choose the most consistent and correct digit-level constraints and valid ranges.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Systematically enumerate all valid (b,c) pairs with b,c in [0,9] satisfying b + c ≤ 8. "
        "Ensure exhaustive enumeration without premature restrictions, facilitating subsequent computations.")

    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    max_round_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(max_round_3_1)]
    all_answer_3_1 = [[] for _ in range(max_round_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(max_round_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking3_1, answer3_1 = await agent([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking3_1, answer3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating (b,c) pairs, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
            all_thinking_3_1[r].append(thinking3_1)
            all_answer_3_1[r].append(answer3_1)

    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2] + all_thinking_3_1[-1] + all_answer_3_1[-1], 
                                                            "Sub-task 4.1: Given all enumerations, provide the final list/count of valid (b,c) pairs.", 
                                                            is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 2: For each valid (b,c) pair from Sub-task 3.1, compute dependent digits d = 1 + b + c and f = 9 - c. "
        "Then enumerate all (a,e) pairs with digits in [0,9] satisfying 10a + 11b + 10c + e = 89. "
        "Verify digit bounds and count all valid solutions, ensuring completeness and correctness.")

    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    max_round_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(max_round_3_2)]
    all_answer_3_2 = [[] for _ in range(max_round_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(max_round_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking3_2, answer3_2 = await agent([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos_3_2 = [taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking3_2, answer3_2 = await agent(input_infos_3_2, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating (a,e) pairs and counting solutions, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
            all_thinking_3_2[r].append(thinking3_2)
            all_answer_3_2[r].append(answer3_2)

    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1] + all_thinking_3_2[-1] + all_answer_3_2[-1], 
                                                            "Sub-task 4.2: Given all enumerations, provide the total count of valid digit assignments.", 
                                                            is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    reflect_instruction_3_3 = (
        "Sub-task 3: Perform reflexion and verification to cross-check the total count of valid digit assignments against the combinatorial sum ∑_{b=0}^8 (9 - b) = 45 and the problem's example. "
        "Detect undercounting or overcounting errors early by reconciling enumeration results with logical bounds and known solutions before finalizing the answer. "
        "Use insights from previous subtasks and feedback to improve accuracy.")

    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    max_round_3_3 = self.max_round
    cot_inputs_3_3 = [taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2, thinking3_1, answer3_1, thinking3_2, answer3_2]
    subtask_desc_3_3 = {
        "subtask_id": "stage_3.subtask_3",
        "instruction": "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better." + reflect_instruction_3_3,
        "context": ["user query", "thinking and answer of all previous subtasks"],
        "agent_collaboration": "Reflexion"
    }

    thinking3_3, answer3_3 = await cot_agent_3_3(cot_inputs_3_3, subtask_desc_3_3['instruction'], 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, verifying total count, thinking: {thinking3_3.content}; answer: {answer3_3.content}")

    critic_inst_3_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(max_round_3_3):
        feedback, correct = await critic_agent_3_3(cot_inputs_3_3 + [thinking3_3, answer3_3], critic_inst_3_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_3_3.extend([thinking3_3, answer3_3, feedback])
        thinking3_3, answer3_3 = await cot_agent_3_3(cot_inputs_3_3, subtask_desc_3_3['instruction'], i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining verification, thinking: {thinking3_3.content}; answer: {answer3_3.content}")

    sub_tasks.append(f"Sub-task 4.3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}")
    subtask_desc_3_3['response'] = {"thinking": thinking3_3, "answer": answer3_3}
    logs.append(subtask_desc_3_3)
    print("Step 3.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_3, answer3_3, sub_tasks, agents)
    return final_answer, logs
