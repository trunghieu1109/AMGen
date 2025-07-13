async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Compute the symbolic volume V of tetrahedron ABCD via the Cayley–Menger determinant; ensure each step is detailed to avoid unnoticed sign or factor errors."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1_i, answer1_i = await cot_agents[i]([taskInfo], cot_sc_instruction, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing symbolic volume, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings.append(thinking1_i)
        possible_answers.append(answer1_i)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent answer for computing the symbolic volume. Given all the above thinking and answers, find the most consistent and correct symbolic volume.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 4: Compute the areas of faces ABC, ABD, ACD, and BCD symbolically via Heron's formula; document each formula application to avoid algebraic slip-ups."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking4_i, answer4_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing symbolic face areas, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings2.append(thinking4_i)
        possible_answers2.append(answer4_i)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 4: Synthesize and choose the most consistent answer for face areas. Given all the above thinking and answers, find the most consistent and correct symbolic face areas.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc2['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction3 = "Sub-task 7: Apply the formula r = 3V/ΣS using V and ΣS to derive a symbolic expression for the inradius; track every algebraic step to avoid arithmetic errors."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction3,
        "context": ["user query", thinking1, answer1, thinking4, answer4],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking7_i, answer7_i = await cot_agents3[i]([taskInfo, thinking1, answer1, thinking4, answer4], cot_sc_instruction3, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, computing inradius formula, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
        possible_thinkings3.append(thinking7_i)
        possible_answers3.append(answer7_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1, thinking4, answer4] + possible_thinkings3 + possible_answers3,
        "Sub-task 7: Synthesize and choose the most consistent answer for inradius formula. Given all the above thinking and answers, find the most consistent and correct symbolic inradius expression.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc3['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 10: Simplify the confirmed inradius expression into the form m√n/p with gcd(m,p)=1 and n squarefree; extract m, n, p and compute m+n+p."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction4,
        "context": ["user query", thinking7, answer7],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent4([taskInfo, thinking7, answer7], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, simplifying inradius expression, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc4['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs