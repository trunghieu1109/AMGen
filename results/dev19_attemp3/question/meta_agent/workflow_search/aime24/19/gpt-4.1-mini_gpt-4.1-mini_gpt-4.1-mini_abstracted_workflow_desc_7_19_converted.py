async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Rewrite and simplify the expression inside the product, i.e., transform 2 - 2*omega^k + omega^(2k) "
        "into a factorized polynomial form in omega^k. This includes verifying the factorization rigorously."
    )
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing expression factorization, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, 
                                                    "Sub-task 1: Synthesize and choose the most consistent factorization of the expression.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Express the product over k=0 to 12 of the simplified polynomial evaluated at omega^k as a resultant or product involving the 13th cyclotomic polynomial Phi_13(x). "
        "Validate the relationship carefully, ensuring inclusion of omega^0=1 and confirming polynomial identities."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, relating product to cyclotomic polynomial, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, 
                                                    "Sub-task 2: Synthesize and choose the most consistent expression relating the product to Phi_13.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Derive the explicit formula for the product as Phi_13(1 + i) * Phi_13(1 - i), by applying the identity product_{k=0}^{12} (t - omega^k) = t^{13} - 1. "
        "Explicitly write out polynomial forms and substitution steps for numeric evaluation."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, deriving explicit formula, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_thinkings_3 + possible_answers_3, 
                                                    "Sub-task 3: Synthesize and finalize explicit formula for product.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_4a = (
        "Sub-task 4a: Compute (1 + i)^{13} and (1 - i)^{13} precisely using polar form, explicitly tracking all factors including negative signs. "
        "Show all algebraic and trigonometric steps in detail. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4a = self.max_round
    all_thinking_4a = [[] for _ in range(N_max_4a)]
    all_answer_4a = [[] for _ in range(N_max_4a)]
    subtask_desc_4a = {
        "subtask_id": "subtask_4a",
        "instruction": debate_instr_4a,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4a):
        for i, agent in enumerate(debate_agents_4a):
            if r == 0:
                thinking4a, answer4a = await agent([taskInfo, thinking3, answer3], debate_instr_4a, r, is_sub_task=True)
            else:
                input_infos_4a = [taskInfo, thinking3, answer3] + all_thinking_4a[r-1] + all_answer_4a[r-1]
                thinking4a, answer4a = await agent(input_infos_4a, debate_instr_4a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing powers, thinking: {thinking4a.content}; answer: {answer4a.content}")
            all_thinking_4a[r].append(thinking4a)
            all_answer_4a[r].append(answer4a)
    final_decision_agent_4a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4a, answer4a = await final_decision_agent_4a([taskInfo, thinking3, answer3] + all_thinking_4a[-1] + all_answer_4a[-1], 
                                                        "Sub-task 4a: Finalize computation of (1+i)^13 and (1-i)^13.", 
                                                        is_sub_task=True)
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc_4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc_4a)
    print("Step 4a: ", sub_tasks[-1])

    debate_instr_4b = (
        "Sub-task 4b: Evaluate Phi_13(1 + i) and Phi_13(1 - i) using Phi_13(t) = (t^{13} - 1)/(t - 1), substituting results from 4a. "
        "Perform division and simplification explicitly, cross-validate results. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4b = self.max_round
    all_thinking_4b = [[] for _ in range(N_max_4b)]
    all_answer_4b = [[] for _ in range(N_max_4b)]
    subtask_desc_4b = {
        "subtask_id": "subtask_4b",
        "instruction": debate_instr_4b,
        "context": ["user query", thinking4a.content, answer4a.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4b):
        for i, agent in enumerate(debate_agents_4b):
            if r == 0:
                thinking4b, answer4b = await agent([taskInfo, thinking4a, answer4a], debate_instr_4b, r, is_sub_task=True)
            else:
                input_infos_4b = [taskInfo, thinking4a, answer4a] + all_thinking_4b[r-1] + all_answer_4b[r-1]
                thinking4b, answer4b = await agent(input_infos_4b, debate_instr_4b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating Phi_13, thinking: {thinking4b.content}; answer: {answer4b.content}")
            all_thinking_4b[r].append(thinking4b)
            all_answer_4b[r].append(answer4b)
    final_decision_agent_4b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4b, answer4b = await final_decision_agent_4b([taskInfo, thinking4a, answer4a] + all_thinking_4b[-1] + all_answer_4b[-1], 
                                                        "Sub-task 4b: Finalize evaluation of Phi_13(1+i) and Phi_13(1-i).", 
                                                        is_sub_task=True)
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc_4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc_4b)
    print("Step 4b: ", sub_tasks[-1])

    debate_instr_4c = (
        "Sub-task 4c: Multiply Phi_13(1 + i) and Phi_13(1 - i) to obtain the exact integer value of the product. "
        "Show all algebraic simplifications and verify integer nature. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4c = self.max_round
    all_thinking_4c = [[] for _ in range(N_max_4c)]
    all_answer_4c = [[] for _ in range(N_max_4c)]
    subtask_desc_4c = {
        "subtask_id": "subtask_4c",
        "instruction": debate_instr_4c,
        "context": ["user query", thinking4b.content, answer4b.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4c):
        for i, agent in enumerate(debate_agents_4c):
            if r == 0:
                thinking4c, answer4c = await agent([taskInfo, thinking4b, answer4b], debate_instr_4c, r, is_sub_task=True)
            else:
                input_infos_4c = [taskInfo, thinking4b, answer4b] + all_thinking_4c[r-1] + all_answer_4c[r-1]
                thinking4c, answer4c = await agent(input_infos_4c, debate_instr_4c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, multiplying values, thinking: {thinking4c.content}; answer: {answer4c.content}")
            all_thinking_4c[r].append(thinking4c)
            all_answer_4c[r].append(answer4c)
    final_decision_agent_4c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4c, answer4c = await final_decision_agent_4c([taskInfo, thinking4b, answer4b] + all_thinking_4c[-1] + all_answer_4c[-1], 
                                                        "Sub-task 4c: Finalize multiplication and verify integer result.", 
                                                        is_sub_task=True)
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc_4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc_4c)
    print("Step 4c: ", sub_tasks[-1])

    reflect_instruction_5 = (
        "Sub-task 5: Reduce the computed integer product modulo 1000 to find the remainder when divided by 1000. "
        "Include a brief verification of modular arithmetic correctness. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4c, answer4c]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": reflect_instruction_5,
        "context": ["user query", thinking4c.content, answer4c.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, reducing modulo 1000, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], 
                                                "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining modulo reduction, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
