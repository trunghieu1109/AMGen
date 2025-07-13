async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Explicitly define the domain of the problem by stating the constraints on the base b (b >= 2), "
        "the digits x and y of the two-digit number in base b (1 <= x <= b-1 for the leading digit, 0 <= y <= b-1 for the second digit), "
        "and the positive integer n represented as n = x*b + y. Emphasize that x must be nonzero to ensure exactly two digits and that all variables are integers."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining domain constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the key equation relating the digits and base: the sum of digits x + y equals the square root of n, "
        "where n = x*b + y, i.e., x + y = sqrt(x*b + y). Highlight the nonlinear nature of this equation and discuss its implications for possible values of x, y, and b without attempting to solve it yet."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulating key equation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct formulation of the key equation and its implications."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent answer for key equation." + final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Clarify and confirm the interpretation of the problem conditions, including the necessity of the leading digit x being nonzero, "
        "the integer nature of digits and base, and the requirement that n is a positive integer with exactly two digits in base b. Avoid assumptions beyond standard base representation rules."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, clarifying problem interpretation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: For a given base b, enumerate all possible digit pairs (x,y) respecting the digit constraints (1 <= x <= b-1, 0 <= y <= b-1). "
        "Generate the complete set of candidate pairs without filtering based on the square root condition."
    )
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_4)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, enumerating digit pairs for base b, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, synthesize the enumeration method for digit pairs (x,y) for a given base b."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize enumeration method." + final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: For each candidate digit pair (x,y) generated for base b, verify whether the condition (x + y)^2 = x*b + y holds exactly. "
        "Enforce strict digit constraints and confirm that the right side is a perfect square equal to (x + y)^2. Record all valid pairs that satisfy this condition for the given base. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4] + all_thinking5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying b-eautiful condition, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Given all the above thinking and answers, reason over them carefully and provide a final verified list of valid digit pairs for base b."
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1], "Sub-task 5: Verify and finalize valid digit pairs." + final_instr_5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflect_instruction_7a = (
        "Sub-task 7a: Implement a computational enumeration process iterating over bases b starting from 2 upwards. "
        "For each base, generate all candidate digit pairs (x,y) as per digit constraints, verify the b-eautiful condition (x + y)^2 = x*b + y, "
        "and record the count of valid b-eautiful numbers. This subtask must exhaustively and explicitly enumerate and verify all candidates to avoid overcounting or missing solutions. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_7a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7a = self.max_round
    cot_inputs_7a = [taskInfo, thinking5]
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": reflect_instruction_7a,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7a, answer7a = await cot_agent_7a(cot_inputs_7a, reflect_instruction_7a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7a.id}, enumerating b-eautiful numbers per base, thinking: {thinking7a.content}; answer: {answer7a.content}")
    for i in range(N_max_7a):
        feedback7a, correct7a = await critic_agent_7a([taskInfo, thinking7a], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7a.id}, providing feedback, thinking: {feedback7a.content}; answer: {correct7a.content}")
        if correct7a.content == "True":
            break
        cot_inputs_7a.extend([thinking7a, feedback7a])
        thinking7a, answer7a = await cot_agent_7a(cot_inputs_7a, reflect_instruction_7a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7a.id}, refining enumeration, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])

    debate_instruction_7b = (
        "Sub-task 7b: Cross-validate the enumeration results from Sub-task 7a by independently verifying the counts of b-eautiful numbers per base. "
        "Resolve any discrepancies or conflicts in counts through a verification process that strictly enforces digit constraints and the perfect square condition. "
        "Document the final verified counts for each base. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_7b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_7b = self.max_round
    all_thinking7b = [[] for _ in range(N_max_7b)]
    all_answer7b = [[] for _ in range(N_max_7b)]
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": debate_instruction_7b,
        "context": ["user query", thinking7a.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7b):
        for i, agent in enumerate(debate_agents_7b):
            if r == 0:
                thinking7b, answer7b = await agent([taskInfo, thinking7a], debate_instruction_7b, r, is_sub_task=True)
            else:
                input_infos_7b = [taskInfo, thinking7a] + all_thinking7b[r-1]
                thinking7b, answer7b = await agent(input_infos_7b, debate_instruction_7b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating counts, thinking: {thinking7b.content}; answer: {answer7b.content}")
            all_thinking7b[r].append(thinking7b)
            all_answer7b[r].append(answer7b)
    final_decision_agent_7b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_7b = "Given all the above thinking and answers, reason over them carefully and provide a final verified count of b-eautiful numbers per base."
    thinking7b, answer7b = await final_decision_agent_7b([taskInfo] + all_thinking7b[-1], "Sub-task 7b: Final verification of counts." + final_instr_7b, is_sub_task=True)
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])

    debate_instruction_8 = (
        "Sub-task 8: Identify the smallest base b >= 2 for which the count of b-eautiful numbers exceeds ten, based on the verified counts from previous subtasks. "
        "Provide a clear and unambiguous conclusion supported by the enumeration and verification data. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": debate_instruction_8,
        "context": ["user query", thinking7b.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7b], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7b] + all_thinking8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying minimal base, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_8 = "Given all the above thinking and answers, reason over them carefully and provide the smallest base b >= 2 with more than ten b-eautiful numbers."
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1], "Sub-task 8: Identify minimal base." + final_instr_8, is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_9 = (
        "Sub-task 9: Perform a final validation step by comparing computed results with known examples, such as verifying that 81 is 13-eautiful, "
        "to calibrate and confirm the correctness of the enumeration and identification process. Address any inconsistencies and finalize the solution."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, validating known examples, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
