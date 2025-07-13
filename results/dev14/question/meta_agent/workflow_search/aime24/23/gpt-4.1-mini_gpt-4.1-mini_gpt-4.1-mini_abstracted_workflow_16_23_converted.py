async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Clarify problem assumptions regarding leading zeros and digit repetition using Debate and Reflexion
    debate_instr_0 = (
        "Sub-task 0: Clarify the problem assumptions regarding leading zeros and digit repetition. "
        "Explicitly determine whether leading zeros are allowed in the 3-digit row numbers and the 2-digit column numbers. "
        "Confirm that digits can be repeated and are in the range 0-9. Avoid proceeding without resolving these ambiguities, as they critically affect the constraints and solution count. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_0 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc_0 = {
        "subtask_id": "subtask_0",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking_0, answer_0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1]
                thinking_0, answer_0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, clarifying assumptions, thinking: {thinking_0.content}; answer: {answer_0.content}")
            all_thinking_0[r].append(thinking_0)
            all_answer_0[r].append(answer_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_0 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_0, answer_0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1], "Sub-task 0: Clarify assumptions." + final_instr_0, is_sub_task=True)
    agents.append(f"Final Decision agent, clarifying assumptions, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    # Stage 1: Define variables and write down sum equations explicitly using Chain-of-Thought (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Define variables a,b,c (top row) and d,e,f (bottom row), each 0-9. "
        "Write the two key sum equations explicitly: (100a + 10b + c) + (100d + 10e + f) = 999 and "
        "(10a + d) + (10b + e) + (10c + f) = 99. Incorporate assumptions about leading zeros and digit repetition clarified in subtask_0. "
        "Avoid simplifying or ignoring any part of the addition at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query", thinking_0],
        "agent_collaboration": "CoT"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo, thinking_0], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining variables and equations, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1 continued: Introduce carry variables for row sums with CoT and Reflexion
    cot_reflect_instruction_2 = (
        "Sub-task 2: Introduce carry variables c1 and c2 for addition of two 3-digit row numbers. "
        "Formulate digit-wise addition equations incorporating these carries, ensuring sum of digits plus carry matches digits of 999. "
        "Provide detailed derivations and examples illustrating carry effects. Avoid ignoring carry effects or assuming digit sums equal target digits directly."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", thinking_1],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1], cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, introducing carries for row sums, thinking: {thinking_2.content}; answer: {answer_2.content}")
    for i in range(self.max_round):
        feedback_2, correct_2 = await critic_agent_2([taskInfo, thinking_2],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback_2.content}; correct: {correct_2.content}")
        if correct_2.content == "True":
            break
        thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1, thinking_2, feedback_2], cot_reflect_instruction_2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining carries for row sums, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1 continued: Introduce carry variables for column sums with CoT and Reflexion
    cot_reflect_instruction_3 = (
        "Sub-task 3: Introduce carry variables c3 and c4 for addition of three 2-digit column numbers. "
        "Formulate digit-wise addition equations incorporating these carries, ensuring sum of digits plus carry matches digits of 99. "
        "Provide examples and clarify how these carries constrain digits. Avoid oversimplifying or ignoring carry effects."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking_1],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3([taskInfo, thinking_1], cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, introducing carries for column sums, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(self.max_round):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback_3.content}; correct: {correct_3.content}")
        if correct_3.content == "True":
            break
        thinking_3, answer_3 = await cot_agent_3([taskInfo, thinking_1, thinking_3, feedback_3], cot_reflect_instruction_3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining carries for column sums, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1 continued: Combine digit-wise addition equations into system of linear Diophantine equations using Self-Consistency CoT
    cot_sc_instruction_4 = (
        "Sub-task 4: Combine digit-wise addition equations from subtasks 2 and 3 to derive a complete system of linear Diophantine equations involving digits a,b,c,d,e,f and carry variables c1,c2,c3,c4. "
        "Analyze these equations to deduce constraints on possible values of digits and carries, including permissible ranges and relationships. "
        "Avoid premature enumeration or assumptions without considering carry constraints."
    )
    N_sc_4 = self.max_sc
    cot_agents_4 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_4)
    ]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking_2, thinking_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking_4, answer_4 = await cot_agents_4[i]([taskInfo, thinking_2, thinking_3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, combining equations, thinking: {thinking_4.content}; answer: {answer_4.content}")
        possible_answers_4.append(answer_4)
        possible_thinkings_4.append(thinking_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent and correct system of equations.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing system of equations, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 1 continued: Perform consistency checks and validate derived constraints using Reflexion and Debate
    reflect_instr_5 = (
        "Sub-task 5: Perform consistency checks and validate derived constraints by testing edge cases and verifying the system models the example and plausible scenarios. "
        "Reflect on whether assumptions about leading zeros and carries align with problem statement and example. "
        "Avoid proceeding to enumeration without confirming correctness and completeness of the model. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": reflect_instr_5,
        "context": ["user query", thinking_4],
        "agent_collaboration": "Reflexion | Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4], reflect_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4] + all_thinking_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, reflect_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating constraints, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo, thinking_4] + all_thinking_5[-1], "Sub-task 5: Final validation and reflection." , is_sub_task=True)
    agents.append(f"Final Decision agent, validating constraints, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Enumerate all possible digit and carry assignments satisfying the system using SC_CoT
    cot_sc_instruction_6 = (
        "Sub-task 6: Enumerate all possible digit assignments (a,b,c,d,e,f) and carry variables (c1,c2,c3,c4) within their domains (digits 0-9, carries 0 or 1) that satisfy the full system of equations derived. "
        "Use systematic search with pruning based on constraints to efficiently explore solution space. Avoid ignoring any constraints or carry conditions."
    )
    N_sc_6 = self.max_sc
    cot_agents_6 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_6)
    ]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking_5],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_6):
        thinking_6, answer_6 = await cot_agents_6[i]([taskInfo, thinking_5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, enumerating solutions, thinking: {thinking_6.content}; answer: {answer_6.content}")
        possible_answers_6.append(answer_6)
        possible_thinkings_6.append(thinking_6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_6, answer_6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 6: Synthesize enumeration results.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing enumeration, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 2 continued: Count total number of valid digit assignments using CoT
    cot_instruction_7 = (
        "Sub-task 7: Count total number of valid digit assignments found in enumeration that satisfy all sum constraints and carry conditions. "
        "Ensure no duplicates and all constraints are respected. Provide clear final answer and verify against example and known partial results. Avoid double counting or omission."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking_6],
        "agent_collaboration": "CoT"
    }
    thinking_7, answer_7 = await cot_agent_7([taskInfo, thinking_6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, counting valid solutions, thinking: {thinking_7.content}; answer: {answer_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_7.content}; answer - {answer_7.content}")
    subtask_desc_7['response'] = {"thinking": thinking_7, "answer": answer_7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_7, answer_7, sub_tasks, agents)
    return final_answer, logs
