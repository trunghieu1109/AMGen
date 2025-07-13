async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage1_sub1 = "Sub-task 1: Precisely establish the geometric configuration of rectangles ABCD and EFGH, including orientation, relative positioning, and labeling of points, ensuring the collinearity of points D, E, C, and F is clearly defined with their order on the line. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1_sub1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1_sub1 = self.max_round
    all_thinking_stage1_sub1 = [[] for _ in range(N_max_stage1_sub1)]
    all_answer_stage1_sub1 = [[] for _ in range(N_max_stage1_sub1)]
    subtask_desc_stage1_sub1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1_sub1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1_sub1):
        for i, agent in enumerate(debate_agents_stage1_sub1):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr_stage1_sub1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking_stage1_sub1[r-1] + all_answer_stage1_sub1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage1_sub1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage1_sub1[r].append(thinking)
            all_answer_stage1_sub1[r].append(answer)
    final_decision_agent_stage1_sub1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_stage1_sub1, answer_stage1_sub1 = await final_decision_agent_stage1_sub1([taskInfo] + all_thinking_stage1_sub1[-1] + all_answer_stage1_sub1[-1], "Sub-task 1: Precisely establish the geometric configuration of rectangles ABCD and EFGH, including orientation, relative positioning, and labeling of points, ensuring the collinearity of points D, E, C, and F is clearly defined with their order on the line. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_stage1_sub1.content}; answer - {answer_stage1_sub1.content}")
    subtask_desc_stage1_sub1['response'] = {"thinking": thinking_stage1_sub1, "answer": answer_stage1_sub1}
    logs.append(subtask_desc_stage1_sub1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_stage1_sub2 = "Sub-task 2: Analyze and formalize the properties of rectangles ABCD and EFGH using given side lengths, confirming side orientations and lengths, and deduce coordinates or vector representations for all vertices consistent with the established configuration. Based on the output from Sub-task 1, consider/calculate potential cases of rectangle vertex coordinates and orientations."
    N_sc = self.max_sc
    cot_agents_stage1_sub2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage1_sub2 = []
    possible_thinkings_stage1_sub2 = []
    subtask_desc_stage1_sub2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage1_sub2,
        "context": ["user query", thinking_stage1_sub1, answer_stage1_sub1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_stage1_sub2[i]([taskInfo, thinking_stage1_sub1, answer_stage1_sub1], cot_sc_instruction_stage1_sub2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_sub2[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_stage1_sub2.append(answer)
        possible_thinkings_stage1_sub2.append(thinking)
    final_decision_agent_stage1_sub2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_stage1_sub2, answer_stage1_sub2 = await final_decision_agent_stage1_sub2([taskInfo, thinking_stage1_sub1, answer_stage1_sub1] + possible_thinkings_stage1_sub2 + possible_answers_stage1_sub2, "Sub-task 2: Synthesize and choose the most consistent and correct solutions for rectangle vertex coordinates and orientations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_stage1_sub2.content}; answer - {answer_stage1_sub2.content}")
    subtask_desc_stage1_sub2['response'] = {"thinking": thinking_stage1_sub2, "answer": answer_stage1_sub2}
    logs.append(subtask_desc_stage1_sub2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_stage2_sub1 = "Sub-task 1: Incorporate the concyclicity condition of points A, D, H, and G by deriving the circle equation or relevant geometric constraints, ensuring the circular condition is correctly integrated with the rectangle configurations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage2_sub1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage2_sub1 = self.max_round
    all_thinking_stage2_sub1 = [[] for _ in range(N_max_stage2_sub1)]
    all_answer_stage2_sub1 = [[] for _ in range(N_max_stage2_sub1)]
    subtask_desc_stage2_sub1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_stage2_sub1,
        "context": ["user query", thinking_stage1_sub2, answer_stage1_sub2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage2_sub1):
        for i, agent in enumerate(debate_agents_stage2_sub1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_stage1_sub2, answer_stage1_sub2], debate_instr_stage2_sub1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_stage1_sub2, answer_stage1_sub2] + all_thinking_stage2_sub1[r-1] + all_answer_stage2_sub1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage2_sub1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage2_sub1[r].append(thinking)
            all_answer_stage2_sub1[r].append(answer)
    final_decision_agent_stage2_sub1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_stage2_sub1, answer_stage2_sub1 = await final_decision_agent_stage2_sub1([taskInfo, thinking_stage1_sub2, answer_stage1_sub2] + all_thinking_stage2_sub1[-1] + all_answer_stage2_sub1[-1], "Sub-task 1: Incorporate the concyclicity condition of points A, D, H, and G by deriving the circle equation or relevant geometric constraints. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_stage2_sub1.content}; answer - {answer_stage2_sub1.content}")
    subtask_desc_stage2_sub1['response'] = {"thinking": thinking_stage2_sub1, "answer": answer_stage2_sub1}
    logs.append(subtask_desc_stage2_sub1)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_stage2_sub2 = "Sub-task 2: Express the collinearity condition of points D, E, C, and F algebraically, and relate segment CE to known lengths and coordinates, carefully verifying the order and relative positions to avoid errors in segment length calculations. Based on the output from Sub-task 2 of Stage 1, consider/calculate potential algebraic expressions and segment lengths."
    cot_agents_stage2_sub2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage2_sub2 = []
    possible_thinkings_stage2_sub2 = []
    subtask_desc_stage2_sub2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_stage2_sub2,
        "context": ["user query", thinking_stage1_sub2, answer_stage1_sub2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_stage2_sub2[i]([taskInfo, thinking_stage1_sub2, answer_stage1_sub2], cot_sc_instruction_stage2_sub2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2_sub2[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_stage2_sub2.append(answer)
        possible_thinkings_stage2_sub2.append(thinking)
    final_decision_agent_stage2_sub2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_stage2_sub2, answer_stage2_sub2 = await final_decision_agent_stage2_sub2([taskInfo, thinking_stage1_sub2, answer_stage1_sub2] + possible_thinkings_stage2_sub2 + possible_answers_stage2_sub2, "Sub-task 2: Synthesize and choose the most consistent and correct algebraic expressions and segment length relations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_stage2_sub2.content}; answer - {answer_stage2_sub2.content}")
    subtask_desc_stage2_sub2['response'] = {"thinking": thinking_stage2_sub2, "answer": answer_stage2_sub2}
    logs.append(subtask_desc_stage2_sub2)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_stage3_sub1 = "Sub-task 1: Combine all geometric and algebraic constraints from previous subtasks to set up a solvable system of equations for the unknown length CE, ensuring no contradictions or overlooked conditions. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage3_sub1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage3_sub1 = self.max_round
    all_thinking_stage3_sub1 = [[] for _ in range(N_max_stage3_sub1)]
    all_answer_stage3_sub1 = [[] for _ in range(N_max_stage3_sub1)]
    subtask_desc_stage3_sub1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3_sub1,
        "context": ["user query", thinking_stage2_sub1, answer_stage2_sub1, thinking_stage2_sub2, answer_stage2_sub2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage3_sub1):
        for i, agent in enumerate(debate_agents_stage3_sub1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_stage2_sub1, answer_stage2_sub1, thinking_stage2_sub2, answer_stage2_sub2], debate_instr_stage3_sub1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_stage2_sub1, answer_stage2_sub1, thinking_stage2_sub2, answer_stage2_sub2] + all_thinking_stage3_sub1[r-1] + all_answer_stage3_sub1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_stage3_sub1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_stage3_sub1[r].append(thinking)
            all_answer_stage3_sub1[r].append(answer)
    final_decision_agent_stage3_sub1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_stage3_sub1, answer_stage3_sub1 = await final_decision_agent_stage3_sub1([taskInfo, thinking_stage2_sub1, answer_stage2_sub1, thinking_stage2_sub2, answer_stage2_sub2] + all_thinking_stage3_sub1[-1] + all_answer_stage3_sub1[-1], "Sub-task 1: Combine all constraints to set up solvable system for CE. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_stage3_sub1.content}; answer - {answer_stage3_sub1.content}")
    subtask_desc_stage3_sub1['response'] = {"thinking": thinking_stage3_sub1, "answer": answer_stage3_sub1}
    logs.append(subtask_desc_stage3_sub1)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_stage3_sub2 = "Sub-task 2: Solve the system of equations accurately to find the length of segment CE, verifying the solution's consistency with all given conditions and geometric properties. Based on the output from Sub-task 1 of Stage 3, consider/calculate the exact length of CE."
    cot_agents_stage3_sub2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage3_sub2 = []
    possible_thinkings_stage3_sub2 = []
    subtask_desc_stage3_sub2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_sc_instruction_stage3_sub2,
        "context": ["user query", thinking_stage3_sub1, answer_stage3_sub1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_stage3_sub2[i]([taskInfo, thinking_stage3_sub1, answer_stage3_sub1], cot_sc_instruction_stage3_sub2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage3_sub2[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_stage3_sub2.append(answer)
        possible_thinkings_stage3_sub2.append(thinking)
    final_decision_agent_stage3_sub2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_stage3_sub2, answer_stage3_sub2 = await final_decision_agent_stage3_sub2([taskInfo, thinking_stage3_sub1, answer_stage3_sub1] + possible_thinkings_stage3_sub2 + possible_answers_stage3_sub2, "Sub-task 2: Synthesize and choose the most consistent and correct solution for length CE.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_stage3_sub2.content}; answer - {answer_stage3_sub2.content}")
    subtask_desc_stage3_sub2['response'] = {"thinking": thinking_stage3_sub2, "answer": answer_stage3_sub2}
    logs.append(subtask_desc_stage3_sub2)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_stage3_sub2, answer_stage3_sub2, sub_tasks, agents)
    return final_answer, logs
