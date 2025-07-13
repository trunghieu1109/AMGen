async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Data Clarification and Restatement

    # Sub-task 1: Identify and clearly state total residents and ownership counts
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state the total number of residents and the ownership counts "
        "of each individual item: diamond rings (195), golf clubs (367), garden spades (562), and candy hearts (900, universal)."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, clarifying total residents and ownership counts, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Clarify and restate counts of residents owning exactly two and exactly three items
    cot_instruction_0_2 = (
        "Sub-task 2: Clarify and restate the counts of residents owning exactly two items (437) and exactly three items (234), "
        "emphasizing the meaning of 'exactly' in the context of four sets."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, clarifying exact ownership counts, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Sub-task 3: Establish universal set context (all own candy hearts)
    cot_instruction_0_3 = (
        "Sub-task 3: Establish the universal set context by noting that all 900 residents own candy hearts, "
        "simplifying the problem to intersections involving candy hearts and the other three items."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, establishing universal set context, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Variable Definition, Equation Formulation, and Solving

    # Sub-task 1: Define variables for unknown intersections and express relationships (Reflexion)
    reflect_inst_1_1 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to define variables for the unknown intersection counts among the three non-universal items and their intersections with candy hearts, including the number owning all four items, and express the relationships between these variables."
    )
    cot_reflect_instruction_1_1 = "Sub-task 1: Define variables and relationships for unknown intersections." + reflect_inst_1_1
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_1 = self.max_round
    cot_inputs_1_1 = [taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_2, answer_0_2, thinking_0_3, answer_0_3],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, defining variables and relationships, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(N_max_1_1):
        feedback_1_1, correct_1_1 = await critic_agent_1_1([taskInfo, thinking_1_1],
            "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback_1_1.content}; answer: {correct_1_1.content}")
        if correct_1_1.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, feedback_1_1])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining variables and relationships, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Sub-task 2: Formulate equations based on exactly two and exactly three items owned (CoT)
    cot_instruction_1_2 = (
        "Sub-task 2: Formulate equations based on the given counts of exactly two and exactly three items owned, "
        "using inclusion-exclusion principles and the defined variables to relate these counts to the unknown intersections."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": [taskInfo, thinking_1_1, answer_1_1],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, formulating equations, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Sub-task 3: Solve system of equations to compute number owning all four items (Reflexion)
    reflect_inst_1_3 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the system of equations derived from the ownership counts to compute the number of residents owning all four items."
    )
    cot_reflect_instruction_1_3 = "Sub-task 3: Solve system of equations for number owning all four items." + reflect_inst_1_3
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, solving equations, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3],
            "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining solution, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Verification and Finalization

    # Sub-task 1: Combine computed intersection values and verify consistency (SC_CoT)
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Combine the computed intersection values and verify consistency with the total population and given ownership counts "
        "to finalize the number of residents owning all four items."
    )
    N_sc = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3, answer_1_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, verifying and finalizing, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_instr_2_1 = "Given all the above thinking and answers, find the most consistent and correct solution for the number of residents owning all four items."
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 2.1: Final synthesis." + final_instr_2_1, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
