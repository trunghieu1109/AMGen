async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive parametric representation of segment AB with parameter t in (0,1), "
        "expressing coordinates X(t), Y(t). Represent family F of unit segments PQ with P=(x,0), Q=(0,y), x,y>0, x^2 + y^2 = 1. "
        "Formulate exact coverage condition for point C=(X,Y) on AB to lie on segment PQ in F: there exist x,y with x^2 + y^2=1, 0 < X/x < 1, 0 < Y/y < 1, and X/x + Y/y = 1. "
        "Validate all representations and domain constraints rigorously, avoiding oversimplifications."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving parametric forms and coverage condition, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc0)

    cot_instruction_1 = (
        "Sub-task 2: Using the coverage condition from Sub-task 1, define function f(x) = X/x + Y/\u221a(1 - x^2) - 1 for x in (0,1), where X,Y depend on parameter t on AB. "
        "Impose tangency (double-root) condition by solving f(x) = 0 and f'(x) = 0 simultaneously. "
        "Derive explicit equations, solve for unique t in (0,1) corresponding to point C on AB not covered by any other segment in F except AB. "
        "Verify uniqueness and that C is distinct from A and B. Avoid oversimplified ratio equalities."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo, thinking0, answer0], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, solving double-root condition for unique t, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)

    reflexion_instruction_2 = (
        "Sub-task 3: Verify the solution t and corresponding point C from Sub-task 2. "
        "Substitute t into parametric AB to get C=(X,Y). Confirm f(x) has a double root at corresponding x (f(x)=0 and f'(x)=0). "
        "Check no other x in (0,1) satisfies f(x)=0 to ensure uniqueness. Confirm C lies strictly inside AB. "
        "Document verification results rigorously."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_3",
        "instruction": reflexion_instruction_2,
        "context": ["user query", thinking0.content, answer0.content, thinking1.content, answer1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflexion_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, verifying uniqueness and double root, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2],
                                               "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflexion_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining verification, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)

    debate_instruction_3 = (
        "Sub-task 4: Compute OC^2 = X^2 + Y^2 for unique point C from previous subtasks. "
        "Express OC^2 as fraction p/q in lowest terms with p,q positive and relatively prime. "
        "Perform rigorous algebraic simplification and verify correctness. Avoid approximations."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying OC^2, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1],
                                                    "Sub-task 4: Synthesize and finalize simplified fraction p/q for OC^2.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)

    cot_instruction_4 = (
        "Sub-task 5: Combine numerator p and denominator q from Sub-task 4 to compute final answer p + q. "
        "Present final result clearly, verify it matches problem requirements, and provide concise summary emphasizing uniqueness and correctness of C and OC^2. "
        "Confirm all validations support final answer with no contradictions."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, aggregating final result p+q, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
