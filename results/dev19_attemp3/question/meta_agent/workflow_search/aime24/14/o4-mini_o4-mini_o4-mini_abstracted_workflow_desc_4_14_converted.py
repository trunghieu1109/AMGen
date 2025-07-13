async def forward_14(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction = "Sub-task 0: Derive algebraic constraints implied by the rhombus and diagonal bisection at the origin for points A, B, C, D on the hyperbola x^2/20 - y^2/24 = 1: C = -A, D = -B, side equality |A - B| = |A + B| => A·B = 0, and hyperbola conditions for A and B"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc0 = {"subtask_id": "subtask_0", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking0i, answer0i = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving constraints, thinking: {thinking0i.content}; answer: {answer0i.content}")
        possible_thinkings.append(thinking0i)
        possible_answers.append(answer0i)
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr0 = "Given all the above thinking and answers, synthesize the most consistent derivation of C = -A, D = -B, A·B = 0, and hyperbola conditions."
    thinking0, answer0 = await final_decision_agent0([taskInfo] + possible_thinkings + possible_answers, final_instr0, is_sub_task=True)
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction1 = "Sub-task 1: Parametrize points A and B on the hyperbola using (sqrt(20)cosh u, sqrt(24)sinh u) and (sqrt(20)cosh v, sqrt(24)sinh v), and translate A·B=0 into 20 cosh u cosh v - 24 sinh u sinh v = 0 (coth u * coth v = 6/5)." + debate_instr
    debate_agents1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking1 = [[]]
    all_answer1 = [[]]
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": debate_instruction1, "context": ["user query", "thinking of subtask 0", "answer of subtask 0"], "agent_collaboration": "Debate"}
    for i, agent in enumerate(debate_agents1):
        thinking1i, answer1i = await agent([taskInfo, thinking0, answer0], debate_instruction1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, parametrizing orthogonality, thinking: {thinking1i.content}; answer: {answer1i.content}")
        all_thinking1[0].append(thinking1i)
        all_answer1[0].append(answer1i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, reason over them carefully and provide the best translated relation coth u * coth v = 6/5."
    thinking1, answer1 = await final_decision_agent1([taskInfo, thinking0, answer0] + all_thinking1[0] + all_answer1[0], final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    reflect_inst2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to express BD^2 = 4|B|^2 = 4(20 cosh^2 v + 24 sinh^2 v) and simplify using cosh^2 - sinh^2 = 1 to a function of sinh^2 v."
    cot_reflect_instruction2 = "Sub-task 2: Compute BD^2 in terms of v." + reflect_inst2
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs2 = [taskInfo, thinking1, answer1]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_reflect_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Reflexion"}
    thinking2, answer2 = await cot_agent2(cot_inputs2, cot_reflect_instruction2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, computing BD^2, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback2, correct2 = await critic_agent2([taskInfo, thinking2, answer2], "Please review the answer above and criticize where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent2.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content == "True":
            break
        cot_inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot_agent2(cot_inputs2, cot_reflect_instruction2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refining BD^2 expression, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Find the infimum of BD^2 = 4(20 + 44 sinh^2 v) under the constraint coth u * coth v = 6/5 and v > arccoth(6/5)."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, computing infimum, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs