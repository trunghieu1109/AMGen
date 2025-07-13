async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0 = (
        "Sub-task 1: Formulate the equations representing the total time for Aya's walk plus coffee time based on the given speeds and times. "
        "Express total time as walking time (distance divided by speed) plus coffee time t (converted to hours), for the two scenarios: "
        "at speed s with total time 4 hours, and at speed s + 2 with total time 2 hours 24 minutes. "
        "Convert minutes to hours and set up the two equations with variables s and t. Do not assume values for s or t."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formulating equations, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking_0,
        "answer": answer_0
    }
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_1 = (
        "Sub-task 2: Based on the equations formulated in Sub-task 1, solve the system of two equations to find s (walking speed in km/h) and t (coffee time in minutes). "
        "Perform algebraic manipulation, convert all times consistently to hours or minutes, isolate variables, and verify positive, physically meaningful solutions. "
        "Avoid premature rounding and ensure unit consistency."
    )
    N = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_1, answer_1 = await cot_sc_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, solving system of equations, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_thinking_1, final_answer_1 = await final_decision_agent_1(
        [taskInfo] + possible_answers_1 + possible_thinkings_1,
        "Sub-task 2: Synthesize and choose the most consistent and correct solutions for s and t.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {final_thinking_1.content}; answer - {final_answer_1.content}")
    subtask_desc_1['response'] = {
        "thinking": final_thinking_1,
        "answer": final_answer_1
    }
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_2 = (
        "Sub-task 3: Calculate the total time Aya takes when walking at speed s + 0.5 km/h, using the values of s and t found in Sub-task 2. "
        "Compute walking time as 9 divided by (s + 0.5), convert to minutes, add coffee time t in minutes, and present the total time in minutes. "
        "Verify unit consistency and correctness of arithmetic."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_2,
        "context": ["user query", final_thinking_1.content, final_answer_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, final_thinking_1, final_answer_1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating total time at speed s+0.5, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking_2,
        "answer": answer_2
    }
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2, answer_2, sub_tasks, agents)
    return final_answer, logs
