async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Problem Understanding and Formula Gathering

    # Sub-task 1: Extract and state all given elements and conditions (SC_CoT)
    cot_instruction_1 = (
        "Sub-task 1: Identify and clearly state all given elements and conditions of the problem, "
        "including triangle ABC, points O (circumcenter) and I (incenter), the perpendicularity condition IA perpendicular to OI, "
        "the circumradius R=13, and the inradius r=6. Avoid calculations or assumptions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    N_sc = self.max_sc
    possible_answers_1 = []
    possible_thinkings_1 = []
    for i in range(N_sc):
        thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_1.id}, iteration {i}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent answer for given elements and conditions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Interpret geometric meaning of IA perpendicular to OI (Debate)
    debate_instruction_2 = (
        "Sub-task 2: Interpret the geometric meaning of the perpendicularity condition IA perpendicular to OI, "
        "clarify spatial relationship between points A, I, and O, and how this constrains the triangle's configuration. "
        "Avoid assuming specific coordinates or symmetry without justification. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_rounds_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_rounds_2)]
    all_answer_2 = [[] for _ in range(N_rounds_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1] + all_thinking_2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_2[r].append(thinking2)
            all_answer_2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide an updated interpretation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Recall and list relevant geometric formulas (SC_CoT)
    cot_instruction_3 = (
        "Sub-task 3: Recall and list relevant geometric formulas and relationships involving circumcenter O, incenter I, circumradius R, inradius r, and side lengths of triangle ABC. "
        "Include Euler's formula for OI, Law of Sines, IA = r / sin(A/2), and inradius formula r = 4R sin(A/2) sin(B/2) sin(C/2). Avoid unrelated formulas or assumptions."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent list of relevant formulas.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Derive explicit formula relating IA, r, s, and a (SC_CoT)
    cot_instruction_4 = (
        "Sub-task 4: Derive an explicit equation relating IA, r, semiperimeter s, and side a, using IA^2 = r^2 + (s - a)^2 or equivalently IA = r / sin(A/2). "
        "Clearly connect incenter-vertex distance to side lengths and semiperimeter. Avoid skipping steps or unjustified assumptions."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent explicit formula relating IA, r, s, and a.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Use perpendicularity condition to derive cos B + cos C = 1/2 (Debate)
    debate_instruction_5 = (
        "Sub-task 5: Use the perpendicularity condition IA perpendicular to OI to establish a trigonometric constraint involving angles B and C, "
        "specifically deriving cos B + cos C = 1/2. Avoid assuming symmetry or simplifications. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_rounds_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_rounds_5)]
    all_answer_5 = [[] for _ in range(N_rounds_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking2, thinking3], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking2, thinking3] + all_thinking_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking2, thinking3] + all_thinking_5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide the trigonometric constraint.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Nonlinear System Formulation and Solution

    # Sub-task 6: Formulate and solve nonlinear system (Debate)
    debate_instruction_6 = (
        "Sub-task 6: Formulate the full nonlinear system involving: (1) perpendicularity condition cos B + cos C = 1/2, "
        "(2) inradius formula r = 4R sin(A/2) sin(B/2) sin(C/2), (3) Euler's formula relating OI, R, and r, and (4) angle sum A + B + C = 180°. "
        "Solve rigorously for angles and side lengths without assuming symmetry or approximations. Provide exact or well-justified numeric solutions with error bounds. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_rounds_6 = self.max_round
    all_thinking_6 = [[] for _ in range(N_rounds_6)]
    all_answer_6 = [[] for _ in range(N_rounds_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking4.content, thinking5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4, thinking5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4, thinking5] + all_thinking_6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking_6[r].append(thinking6)
            all_answer_6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo, thinking4, thinking5] + all_thinking_6[-1], "Sub-task 6: Given all the above thinking and answers, reason over them carefully and provide the solution to the nonlinear system.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Express AB * AC in terms of solved angles and known quantities (SC_CoT)
    cot_instruction_7 = (
        "Sub-task 7: Express the product AB * AC in terms of the solved angles and known quantities, using Law of Sines or other relevant formulas. "
        "Prepare expression for numeric evaluation or exact substitution. Avoid premature numeric substitution before verification."
    )
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_7 = []
    possible_thinkings_7 = []
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6], cot_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize and choose the most consistent expression for AB * AC.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Verify and refine computed value of AB * AC (Debate)
    debate_instruction_8 = (
        "Sub-task 8: Verify and refine the computed value of AB * AC by cross-checking all constraints and formulas, including perpendicularity, inradius, circumradius relations, and angle sum. "
        "Challenge assumptions or approximations. Use symbolic algebra or numeric solvers to confirm consistency and justify final numeric or exact answer. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_rounds_8 = self.max_round
    all_thinking_8 = [[] for _ in range(N_rounds_8)]
    all_answer_8 = [[] for _ in range(N_rounds_8)]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": debate_instruction_8,
        "context": ["user query", thinking7.content, thinking6.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7, thinking6], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7, thinking6] + all_thinking_8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking_8[r].append(thinking8)
            all_answer_8[r].append(answer8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo, thinking7, thinking6] + all_thinking_8[-1], "Sub-task 8: Given all the above thinking and answers, reason over them carefully and provide a verified and refined value for AB * AC.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Present final value of AB * AC with justification (SC_CoT)
    cot_instruction_9 = (
        "Sub-task 9: Present the final value of AB * AC in exact form if possible (fractions, surds), or provide clear justification for any decimal approximation. "
        "Critically evaluate plausibility and neatness in context of contest geometry problems. Avoid accepting approximate answers without justification or rounding explanation."
    )
    cot_agents_9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_9 = []
    possible_thinkings_9 = []
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking9, answer9 = await cot_agents_9[i]([taskInfo, thinking8], cot_instruction_9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_9[i].id}, thinking: {thinking9.content}; answer: {answer9.content}")
        possible_answers_9.append(answer9)
        possible_thinkings_9.append(thinking9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + possible_thinkings_9, "Sub-task 9: Synthesize and choose the most consistent and justified final value for AB * AC.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
