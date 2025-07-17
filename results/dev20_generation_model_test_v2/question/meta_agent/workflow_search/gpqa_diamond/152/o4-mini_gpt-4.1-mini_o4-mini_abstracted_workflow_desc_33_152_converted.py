async def forward_152(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Extract reactants, conditions, and explicit major products
    cot_sc_instruction = (
        "Sub-task 1: Extract for reactions A, B, and C the reactants, conditions, "
        "and the exact major final products with explicit tautomeric forms from the question."
    )
    cot_agents = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(self.max_sc)
    ]
    possible_think = []
    possible_ans = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think.append(thinking)
        possible_ans.append(answer)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1(
        [taskInfo] + possible_think + possible_ans,
        "Sub-task 1: Synthesize the most consistent extraction of reactants and final products.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Parse all four multiple-choice options and annotate differences
    cot_sc_instruction2 = (
        "Sub-task 2: Parse each choice (1–4) to extract the systematic names for A, B, and C, "
        "and annotate key differences in ester positions, ring attachments, and tautomeric forms."
    )
    cot_agents2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(self.max_sc)
    ]
    possible_think2 = []
    possible_ans2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents2:
        thinking, answer = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think2.append(thinking)
        possible_ans2.append(answer)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision2(
        [taskInfo, thinking1, answer1] + possible_think2 + possible_ans2,
        "Sub-task 2: Synthesize the most consistent parsing of the four choices.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Mechanistic analysis (nucleophile, electrophile, β-attack)
    reflect_inst3 = (
        "Sub-task 3: Based on the extracted reactants and parsed choices, "
        "perform a detailed mechanistic analysis for each reaction: identify the nucleophile, electrophile, site of β‐attack, "
        "and outline the resulting carbon skeleton before tautomerization. "
        "Given previous attempts and feedback, carefully consider where you could go wrong."
    )
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": reflect_inst3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent3(cot_inputs3, reflect_inst3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for _ in range(self.max_round):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3],
            "Please review the answer above and criticize where it might be wrong. If absolutely correct, output exactly 'True' in 'correct'.", is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content.strip() == "True":
            break
        cot_inputs3 += [thinking3, answer3, feedback3]
        thinking3, answer3 = await cot_agent3(cot_inputs3, reflect_inst3, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Verify tautomeric forms
    reflect_inst4 = (
        "Sub-task 4: Reconcile the mechanistic skeletons with the explicit major final-product tautomeric forms: "
        "confirm that keto vs. enol forms align with the question's wording. "
        "Given previous attempts and feedback, carefully consider where you could go wrong."
    )
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": reflect_inst4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent4(cot_inputs4, reflect_inst4, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for _ in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4],
            "Please review the answer above and criticize where it might be wrong. If absolutely correct, output exactly 'True' in 'correct'.", is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content.strip() == "True":
            break
        cot_inputs4 += [thinking4, answer4, feedback4]
        thinking4, answer4 = await cot_agent4(cot_inputs4, reflect_inst4, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refined thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Debate to match predicted skeletons against choices
    debate_instruction5 = (
        "Sub-task 5: Compare the predicted, tautomer-verified skeletons against the parsed choices. "
        "Use a Debate format to eliminate mismatches on skeleton, numbering, or tautomeric form, and identify the best match for A, B, and C. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice."
    )
    debate_agents5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    all_think5 = [[]]
    all_ans5 = [[]]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        all_think5.append([])
        all_ans5.append([])
        for agent in debate_agents5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4] + all_think5[r] + all_ans5[r]
                thinking5, answer5 = await agent(inputs, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_think5[r].append(thinking5)
            all_ans5[r].append(answer5)
    final_decision5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5(
        [taskInfo, thinking4, answer4] + all_think5[-1] + all_ans5[-1],
        "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Final choice selection with CoT
    cot_instruction6 = (
        "Sub-task 6: Present the final choice number (1–4) with a concise rationale for each product A, B, C, "
        "emphasizing alignment with mechanistic analysis and tautomeric forms."
    )
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6(
        [taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking6, "answer": answer6}
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs