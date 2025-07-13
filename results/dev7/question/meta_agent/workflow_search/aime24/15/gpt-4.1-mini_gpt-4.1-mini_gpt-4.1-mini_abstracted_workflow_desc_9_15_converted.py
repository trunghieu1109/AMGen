async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Aggregate and summarize all given numerical data from the problem statement. "
        "Include total residents, counts of residents owning each individual item (diamond ring, golf clubs, garden spade, candy hearts), "
        "and counts of residents owning exactly two or exactly three items. Explicitly highlight the universal ownership of candy hearts by all 900 residents. "
        "Emphasize the ambiguity regarding whether the 'exactly two' and 'exactly three' counts include candy hearts. "
        "Avoid making untested assumptions about these counts. Prepare the data clearly for subsequent analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, aggregating data, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Assume the counts of residents owning exactly two and exactly three items include candy hearts as one of the items. "
        "Define variables for the number of residents owning exactly one, exactly two, and exactly three of the three items (diamond ring, golf clubs, garden spade). "
        "Use the principle that the sum of the sizes of the sets D, G, and S equals a + 2b + 3c, where a, b, and c correspond to these counts. "
        "Carefully set up the system of equations incorporating the universal candy hearts ownership and the given counts. "
        "Avoid algebraic errors and double counting. Prepare the equations for solving."
    )
    N = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_2, answer_2 = await cot_sc_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, analyzing counts including candy hearts, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and correct solutions assuming candy hearts inclusion." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Assume the counts of residents owning exactly two and exactly three items exclude candy hearts, considering only diamond ring, golf clubs, and garden spade. "
        "Define variables and set up the corresponding system of equations under this assumption. "
        "Compare the structure and implications of these equations with those from Sub-task 2. "
        "Avoid premature conclusions; focus on clear formulation and logical consistency."
    )
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_3, answer_3 = await cot_sc_agents_3[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, analyzing counts excluding candy hearts, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3)
        possible_thinkings_3.append(thinking_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent and correct solutions assuming candy hearts exclusion." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Solve the systems of equations derived in Sub-tasks 2 and 3 to find the number of residents owning all four items under each assumption. "
        "Carefully perform algebraic manipulations, ensuring correct application of inclusion-exclusion principles and counting coefficients. "
        "Document intermediate results and highlight any contradictions or inconsistencies that arise. Avoid ignoring conflicting numeric outcomes."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking_2.content, answer_2.content, thinking_3.content, answer_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_4, answer_4 = await cot_agents_4[i]([taskInfo, thinking_2, answer_2, thinking_3, answer_3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, solving systems and comparing results, thinking: {thinking_4.content}; answer: {answer_4.content}")
        possible_answers_4.append(answer_4)
        possible_thinkings_4.append(thinking_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 4: Synthesize and document contradictions or consistencies in solutions." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_5 = (
        "Sub-task 5: Conduct a debate and critical review of the assumptions, algebraic coefficients, and intermediate results from previous subtasks. "
        "Explicitly challenge the assumption about candy hearts inclusion in the 'exactly two' and 'exactly three' counts. "
        "Evaluate contradictions in numeric results and discuss which interpretation aligns best with the problem constraints and logical reasoning. "
        "Aim to converge on a consistent, justifiable assumption and solution approach."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instr_5,
        "context": ["user query", thinking_4.content, answer_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating assumptions and results, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Conclude on the most consistent assumption and solution." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Stage 2 Sub-task 1: Based on the consensus from the debate, finalize the calculation of the number of residents owning all four items. "
        "Provide a clear, step-by-step explanation of the reasoning, the final numeric answer, and how it satisfies all given data and constraints. "
        "Avoid ambiguity or unsupported conclusions."
    )
    cot_sc_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc_6 = {
        "subtask_id": "stage2_subtask_1",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking_5.content, answer_5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_6, answer_6 = await cot_sc_agents_6[i]([taskInfo, thinking_5, answer_5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_6[i].id}, finalizing answer, thinking: {thinking_6.content}; answer: {answer_6.content}")
        possible_answers_6.append(answer_6)
        possible_thinkings_6.append(thinking_6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_6, answer_6 = await final_decision_agent_6([taskInfo] + possible_answers_6 + possible_thinkings_6, "Stage 2 Sub-task 1: Finalize the number of residents owning all four items." , is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_7 = (
        "Stage 2 Sub-task 2: Perform a thorough verification and reflection on the final answer. "
        "Re-express the problem constraints and confirm that the solution is consistent with total residents, ownership counts, and the universal candy hearts ownership. "
        "Identify and resolve any residual contradictions or uncertainties. Provide a final confirmed answer with detailed justification and highlight lessons learned regarding assumption validation and combinatorial reasoning."
    )
    cot_reflect_instruction_7 = "Stage 2 Sub-task 2: Your problem is to verify and reflect on the final number of residents owning all four items." + reflect_inst_7
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking_6, answer_6]
    subtask_desc_7 = {
        "subtask_id": "stage2_subtask_2",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking_6.content, answer_6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_7, answer_7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying final solution, thinking: {thinking_7.content}; answer: {answer_7.content}")
    for i in range(N_max_7):
        feedback_7, correct_7 = await critic_agent_7([taskInfo, thinking_7, answer_7], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback_7.content}; answer: {correct_7.content}")
        if correct_7.content.strip() == "True":
            break
        cot_inputs_7.extend([thinking_7, answer_7, feedback_7])
        thinking_7, answer_7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining final solution, thinking: {thinking_7.content}; answer: {answer_7.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_7.content}; answer - {answer_7.content}")
    subtask_desc_7['response'] = {"thinking": thinking_7, "answer": answer_7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_7, answer_7, sub_tasks, agents)
    return final_answer, logs
